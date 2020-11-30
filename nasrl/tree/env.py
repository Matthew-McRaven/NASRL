import gym, gym.spaces
import more_itertools
from more_itertools.more import sample
import numpy as np
from numpy.core.numeric import roll
from numpy.lib.arraysetops import isin
import torch

import librl.replay
import librl.nn.core, librl.nn.classifier
import librl.task, librl.task.classification
import librl.train.classification.label

from ..task import *
from .actions import *
class field_idx(enum.Enum):
    type = 0
    channels = 1
    kernel = 2
    stride = 3
    dilation = 4
    padding = 5
    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member
    def __index__(self):
        return self.value
    def __int__(self):
        return self.value

class type_idx(enum.Enum):
    conv = 0
    avg = 1
    max = 2
    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member
    def __index__(self):
        return self.value
    def __int__(self):
        return self.value

# Classification while jointly building CNN's and MLP's
class JointClassificationEnv(gym.Env):
    def __init__(self, data_dim, cnn_count, mlp_count, inner_loss=None, train_data_iter=None, validation_data_iter=None, labels=10):
        super(JointClassificationEnv, self).__init__()
        assert not isinstance(train_data_iter, torch.utils.data.DataLoader) # type: ignore
        assert not isinstance(validation_data_iter, torch.utils.data.DataLoader) # type: ignore

        self.cnn_count = cnn_count
        self.mlp_count = mlp_count
        # Limit the size of a neural network.
        # TODO: Actually respect this limit.
        # type, out_channel, kernel_size, dilation, stride, padding.
        low = np.tile(np.asarray([0,1,1,0,0,0]), (cnn_count,1))
        high = np.tile(np.asarray([2,128,8,8,8,8]), (cnn_count,1))
        self.cnn_observation_space = gym.spaces.Box(low, high, [cnn_count, 6], dtype=np.int16)
        self.mlp_observation_space = gym.spaces.Box(0, 400, (mlp_count,), dtype=np.int16)
        self.observation_space = gym.spaces.Tuple((self.cnn_observation_space, self.mlp_observation_space))
        # TODO: Figure out a sane representation of our action space.
        self.labels = labels
        self.data_dim = data_dim

        self.inner_loss = inner_loss
        self.train_data_iter = train_data_iter
        self.validation_data_iter = validation_data_iter

    # TODO: Generate an observation that isn't just a random array.
    def reset(self):
        self.cnn_state = [librl.nn.core.cnn.conv_def(4, 4, 1, 1, 1)]
        self.mlp_state = self.mlp_observation_space.sample()
        return self.convert_state_numpy(self.cnn_state), self.mlp_state
   
    # Convert a list of convolutional defs to a numpy array.
    def convert_state_numpy(self, state):
        np_state = np.zeros((self.cnn_count, 6))
        channels = self.data_dim[0]
        for idx, layer in enumerate(state):
            if isinstance(layer, librl.nn.core.cnn.conv_def):
                np_state[idx, field_idx.type] = type_idx.conv
                channels = layer.out_channels
            if isinstance(layer, librl.nn.core.cnn.pool_def):
                if layer.pool_type == "avg": np_state[idx, field_idx.type] = type_idx.avg
                elif layer.pool_type == "max": np_state[idx, field_idx.type] = type_idx.max
            np_state[idx, field_idx.channels] = channels
            np_state[idx, field_idx.kernel] = layer.kernel
            np_state[idx, field_idx.stride] = layer.stride
            np_state[idx, field_idx.dilation] = layer.dilation
            np_state[idx, field_idx.padding] = layer.padding
        print(np_state)
        return np_state

    # Check that a given state has a non-zero output size.
    def cnn_check_size(self, state):
        current_dims = list(self.data_dim[1:])
        for item in state:
            for dim, val in enumerate(current_dims):
                current_dims[dim] = librl.nn.core.cnn.resize_convolution(val, item.kernel, item.dilation, item.stride, item.padding)
                if current_dims[dim] <= 0: return False
        return True

    def step(self, actions):
        self.old_state = self.cnn_state

        # Apply each modification in our action array.
        for action in actions:
            # Add neurons by performing a rotate right from the insertion point, and setting insertion point.
            if isinstance(action, ActionAddConv) or isinstance(action, ActionAddPool):
                self.cnn_state.insert(action.layer_num, action.conv_def) 
            elif isinstance(action, ActionAddMLP):
                self.mlp_state[action.layer_num:] = np.roll(self.mlp_state[action.layer_num:], 1)
                self.mlp_state[action.layer_num] = action.layer_size
            elif isinstance(action, ActionDelete) and action.which == LayerType.CNN:
                # Remove neurons by performing a rotate left from the insertion point, and clearing the last element.
                if len(self.cnn_state) == 1: (self.convert_state_numpy(self.cnn_state), self.mlp_state), -1., False, {}
                elif len(self.cnn_state) <= action.layer_num: self.cnn_state.pop(-1)
                else: self.cnn_state.pop(action.layer_num) 
            elif isinstance(action, ActionDelete) and action.which == LayerType.MLP:
                # Remove neurons by performing a rotate left from the insertion point, and clearing the last element.
                if np.count_nonzero(self.mlp_state) == 1: (self.convert_state_numpy(self.cnn_state), self.mlp_state), -1., False, {}
                else:
                    self.mlp_state[action.layer_num:] = np.roll(self.mlp_state[action.layer_num:], -1)
                    self.mlp_state[-1] = 0
            # TODO: Add a no-op action.
            else: raise NotImplementedError("Not a valid action")

        # Require that the new state yield +'ve image size.
        # TODO: Debug if old state is really restored correctly.
        if not self.cnn_check_size(self.cnn_state):
            self.cnn_state = self.cnn_old_state
            return (self.convert_state_numpy(self.cnn_state), self.mlp_state), -1., False, {}
        mlp_state = [x for x in self.mlp_state if x>0]
        #print(mlp_state)

        # Create a classification network using our current state.
        class_kernel = create_nn_from_def(self.data_dim, self.cnn_state, mlp_state)
        class_net = librl.nn.classifier.Classifier(class_kernel, self.labels)
        class_net.train()

        # Create and run a classification task.
        t, v = self.train_data_iter, self.validation_data_iter
        cel = torch.nn.CrossEntropyLoss()
        inner_task = librl.task.classification.ClassificationTask(classifier=class_net, criterion=cel, train_data_iter=t, validation_data_iter=v)
        correct_list, total_list = librl.train.classification.label.train_single_label_classifier([inner_task])
        # Reward is % accuracy on validation set.
        state = (self.convert_state_numpy(self.cnn_state), self.mlp_state)
        return state, 100 * correct_list[0] / total_list[0], False, {}

# Classification where only CNN's will be used.
class CNNClassificationEnv(JointClassificationEnv):
    def __init__(self, data_dim, cnn_count, inner_loss=None, train_data_iter=None, validation_data_iter=None, labels=10):
        super(CNNClassificationEnv, self).__init__(data_dim, cnn_count, 1, inner_loss, train_data_iter, validation_data_iter, labels)
        self.observation_space = self.cnn_observation_space
    def reset(self):
        self.cnn_state = [librl.nn.core.cnn.conv_def(4, 4, 1, 1, 1)]
        self.mlp_state = [200]
        # Only expose the state of the CNN.
        return self.convert_state_numpy(self.cnn_state)
    def step(self, actions):
        for action in actions:
            assert isinstance(action, ActionAddConv) or isinstance(action, ActionAddPool) or (
                isinstance(action, ActionDelete) and action.which == LayerType.CNN)
        state, reward, done, dict = super(CNNClassificationEnv, self).step(actions)
        # Only expose the state of the CNN.
        return state[0], reward, done, dict

# Classification where only MLP's will be used.
class MLPClassificationEnv(JointClassificationEnv):
    def __init__(self, data_dim, linear_count, inner_loss=None, train_data_iter=None, validation_data_iter=None, labels=10):
        super(MLPClassificationEnv, self).__init__(data_dim, 0, linear_count, inner_loss, train_data_iter, validation_data_iter, labels)
        self.observation_space = self.mlp_observation_space

    def reset(self):
        self.cnn_state = []
        self.mlp_state = self.observation_space.sample()
        # Only expose the state of the MLP.
        return self.mlp_state
    def step(self, actions):
        for action in actions:
            print(action)
            assert isinstance(action, ActionAddMLP) or (
                isinstance(action, ActionDelete) and action.which == LayerType.MLP)
        state, reward, done, dict = super(MLPClassificationEnv, self).step(actions)
        # Only expose the state of the MLP.
        return state[1], reward, done, dict