import itertools

import gym, gym.spaces
import more_itertools
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
import nasrl.reward
import nasrl.field
class field_idx(enum.Enum):
    type = 0
    channels = 1
    kernel = 2
    stride = 3
    padding = 4
    dilation = 5

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

    # COmpute min, max needed for spaces.
    @staticmethod
    def extract_shapes(cnn_conf):
        fields = [cnn_conf.channel,cnn_conf.kernel, cnn_conf.stride, cnn_conf.padding, cnn_conf.dilation]
        # Extract min, max from field definitions, and remap to sane values for NN.
        min = map(lambda f: f.transform.forward(f.domain.min),fields)
        max = map(lambda f: f.transform.forward(f.domain.max),fields)
        # TODO: Add term for # of types
        return itertools.chain([0], min), itertools.chain([2], max)

    def __init__(self, data_dim, cnn_conf, mlp_conf, inner_loss=None, train_data_iter=None, 
    validation_data_iter=None, classes=10, reward_fn=nasrl.reward.Linear, adapt_steps=3
    ):
        super(JointClassificationEnv, self).__init__()
        assert not isinstance(train_data_iter, torch.utils.data.DataLoader) # type: ignore
        assert not isinstance(validation_data_iter, torch.utils.data.DataLoader) # type: ignore
        # We must have at least either an MLP or CNN config. If present, require type correctness.
        assert not (mlp_conf is None and cnn_conf is None)
        assert mlp_conf is None or isinstance(mlp_conf, nasrl.field.conf.MLPConf)
        assert cnn_conf is None or isinstance(cnn_conf, nasrl.field.conf.CNNConf)
        
        
        # Create default MLP conf
        if mlp_conf is None:
            mlp_conf = nasrl.field.MLPConf(0, None)
        # Create default CNN conf
        if cnn_conf is None:
            cnn_conf = nasrl.field.CNNConf(0, None, None, None, None, None)
            
        assert classes > 1

        self.cnn_conf = cnn_conf
        self.mlp_conf = mlp_conf

        self.classes = classes
        self.reward_fn = reward_fn
        self.adapt_steps = adapt_steps

        # Limit CNN min / max values by examining CNN config
        if cnn_conf.n_layers > 0:
            # type, out_channel, kernel_size, stride, padding, dilation.
            mins, maxs = JointClassificationEnv.extract_shapes(cnn_conf)
            low = np.tile(np.fromiter(mins, dtype=np.float64, count=6), (cnn_conf.n_layers, 1))
            high = np.tile(np.fromiter(maxs, dtype=np.float64, count=6), (cnn_conf.n_layers, 1))
            self.cnn_observation_space = gym.spaces.Box(low, high, [cnn_conf.n_layers, 6], dtype=np.int16)
        # Limit MLP min / max values by examining MLP config
        if mlp_conf.n_layers > 0:
            min = mlp_conf.mlp.transform.forward(mlp_conf.mlp.domain.min)
            max = mlp_conf.mlp.transform.forward(mlp_conf.mlp.domain.max)
            self.mlp_observation_space = gym.spaces.Box(min, max, (mlp_conf.n_layers,), dtype=np.int16)
        
        # If we are configured to generate CNN's and MLP's, we need a cartesian product to represent our state.
        if mlp_conf.n_layers > 0 and cnn_conf.n_layers > 0:
            self.observation_space = gym.spaces.Tuple((self.cnn_observation_space, self.mlp_observation_space))
        # Else we have either CNN's xor MLP's, in which case we can ignore this missing space.
        elif mlp_conf.n_layers > 0:
            self.observation_space = self.mlp_observation_space
        elif cnn_conf.n_layers > 0:
            self.observation_space = self.cnn_observation_space
        else: raise ValueError("No layers in either CNN or MLP. Probably configuration bug.")

        # TODO: Figure out a sane representation of our action space.
        # Probably some intersection of discrete/box tuples.
        self.data_dim = data_dim

        self.inner_loss = inner_loss
        self.train_data_iter = train_data_iter
        self.validation_data_iter = validation_data_iter

    # Rese current state by sa
    def reset(self):
        # Generate a new CNN if CNN's are allowed.
        if self.cnn_conf.n_layers > 0:
            self.cnn_state = []
            for idx, i in enumerate(range(self.cnn_conf.init_layer_count())):
                self.cnn_state.append(self.cnn_conf.init_layer_value(idx))
        # Generate the field even if not present, to make `step` less painful.
        else: self.cnn_state = []
        # Generate a new MLP if MLP's are allowed.
        if self.mlp_conf.n_layers > 0:
            self.mlp_state = np.zeros((self.mlp_conf.n_layers,), dtype=np.uint16)
            for idx, i in enumerate(range(self.mlp_conf.init_layer_count())):
                self.mlp_state[idx] = int(self.mlp_conf.mlp.init_value(idx))
        # Generate the field even if not present, to make `_get_normalized_state` less painful.
        else: self.mlp_state = np.zeros((0,), dtype=np.uint16)
        # Final sanity check that our generated network has *something* in it. Empty networks are bad.
        assert self.cnn_state or (self.mlp_state.any() > 0)
        return self._get_normalized_state()
   
    # Convert a list of convolutional defs to a numpy array.
    def convert_state_numpy(self, state):
        if self.cnn_conf.n_layers == 0: return None
        np_state = np.zeros((self.cnn_conf.n_layers, 6))
        channels = self.data_dim[0]
        for idx, layer in enumerate(state):
            if isinstance(layer, librl.nn.core.cnn.conv_def):
                np_state[idx, field_idx.type] = type_idx.conv
                channels = layer.out_channels
            if isinstance(layer, librl.nn.core.cnn.pool_def):
                if layer.pool_type == "avg": np_state[idx, field_idx.type] = type_idx.avg
                elif layer.pool_type == "max": np_state[idx, field_idx.type] = type_idx.max
            # Remap human-readable representations of CNN definition to something amenable to a NN.
            np_state[idx, field_idx.channels] = self.cnn_conf.channel.transform.forward(channels)
            np_state[idx, field_idx.kernel] = self.cnn_conf.kernel.transform.forward(layer.kernel)
            np_state[idx, field_idx.stride] = self.cnn_conf.stride.transform.forward(layer.stride)
            np_state[idx, field_idx.padding] = self.cnn_conf.padding.transform.forward(layer.padding)
            np_state[idx, field_idx.dilation] = self.cnn_conf.dilation.transform.forward(layer.dilation)
        return np_state

    # Check that a given state has a non-zero output size.
    def cnn_check_size(self, state):
        current_dims = list(self.data_dim[1:])
        for item in state:
            for dim, val in enumerate(current_dims):
                current_dims[dim] = librl.nn.core.cnn.resize_convolution(val, item.kernel, item.dilation, item.stride, item.padding)
                if current_dims[dim] <= 0: return False
        return True

    # Extract and normalize fields shared between convolutional and pooling layers.
    def _normalize_core(self, action):
        kernel = int(self.cnn_conf.kernel.transform.backward(action.kernel))
        stride = int(self.cnn_conf.stride.transform.backward(action.stride))
        padding = int(self.cnn_conf.padding.transform.backward(action.padding))
        dilation = int(self.cnn_conf.dilation.transform.backward(action.dilation))
        return kernel, stride, padding, dilation

    # Create a convolutional layer definition from an action. 
    def _normalize_conv(self, action) -> librl.nn.core.cnn.conv_def:
        out_channels = int(self.cnn_conf.channel.transform.backward(action.channel))
        kernel, stride, padding, dilation = self._normalize_core(action)
        return librl.nn.core.cnn.conv_def(kernel, out_channels, stride, padding, dilation)

    # Create a pool definition from an action. 
    def _normalize_pool(self, action) -> librl.nn.core.cnn.pool_def:
        kernel, stride, padding, dilation = self._normalize_core(action)
        if action.pool_type == 'avg': dilation == 1
        return librl.nn.core.cnn.pool_def(kernel, stride, padding, dilation, pool_type=action.pool_type)

    # Convert CNN & MLP state to something presentable to a neural net.
    def _get_normalized_state(self):
        cnn_state, mlp_state = None, None
        if self.cnn_conf.n_layers>0: cnn_state = self.convert_state_numpy(self.cnn_state)
        if self.mlp_conf.n_layers>0:
            fn = lambda x: self.mlp_conf.mlp.transform.forward(x)
            mlp_state = fn(self.mlp_state)
        return cnn_state, mlp_state

    # Return True if action(s) were applied succesfully, or False if not.
    # Excepts actions to be iterable.
    def _apply_actions(self, actions):
        # Apply each modification in our action array.
        for action in actions:
            # Add neurons by performing a rotate right from the insertion point, and setting insertion point.
            if isinstance(action, ActionAddConv):
                self.cnn_state.insert(action.layer_num, self._normalize_conv(action))
            elif isinstance(action, ActionAddPool): 
                self.cnn_state.insert(action.layer_num, self._normalize_pool(action))
            elif isinstance(action, ActionAddMLP):
                self.mlp_state[action.layer_num:] = np.roll(self.mlp_state[action.layer_num:], 1)
                # Backward() not guaranteed to return int.
                self.mlp_state[action.layer_num] = int(self.mlp_conf.mlp.transform.backward(action.layer_size))
            elif isinstance(action, ActionDelete) and action.which == LayerType.CNN:
                # Remove neurons by performing a rotate left from the insertion point, and clearing the last element.
                if len(self.cnn_state) <= 1: return False
                elif len(self.cnn_state) <= action.layer_num: self.cnn_state.pop(-1)
                else: self.cnn_state.pop(action.layer_num) 
            elif isinstance(action, ActionDelete) and action.which == LayerType.MLP:
                # Remove neurons by performing a rotate left from the insertion point, and clearing the last element.
                if np.count_nonzero(self.mlp_state) <= 1: return False
                else:
                    self.mlp_state[action.layer_num:] = np.roll(self.mlp_state[action.layer_num:], -1)
                    self.mlp_state[-1] = 0
            # TODO: Add a no-op action.
            else: raise NotImplementedError("Not a valid action")
        return True

    def step(self, actions):
        # Adding a layer may reduce an output dimension to <- 0.
        # Must cache old state so we back out of this failure mode.
        cnn_old_state = self.cnn_state

        # Actions couldn't be applied.
        # Typically occurs because you attempted to delete the last layer.
        if not self._apply_actions(actions):
            # Explicitly log that this is a broken config.
            # TODO: Figure out a better retval.
            return self._get_normalized_state(), -1., False, {'broken':True}

        # Require that the new state have positive value dimensions.
        # TODO: Debug if old state is really restored correctly.
        if self.cnn_conf.n_layers>0 and not self.cnn_check_size(self.cnn_state):
            self.cnn_state = cnn_old_state
            # Explicitly log that this is a broken config.
            return self._get_normalized_state(), -1., False, {'broken':True}

        # Strip 0 layers from MLP state.    
        mlp_state = [x for x in self.mlp_state if x>0]

        # Create a classification network using our current state.
        class_kernel = create_nn_from_def(self.data_dim, self.cnn_state, mlp_state)
        class_net = librl.nn.classifier.Classifier(class_kernel, self.classes)
        print(class_net)
        class_net.train()

        # Create and run a classification task.
        t, v = self.train_data_iter, self.validation_data_iter
        cel = torch.nn.CrossEntropyLoss()
        inner_task = librl.task.classification.ClassificationTask(classifier=class_net, criterion=cel, train_data_iter=t, validation_data_iter=v)
        
        correct, total = [], []
        # TODO: Only perform validation step on `last` step.
        for _ in range(self.adapt_steps + 1):
            inner_correct, inner_total = librl.train.classification.label.train_single_label_classifier([inner_task])
            correct.extend(inner_correct), total.extend(inner_total)

        reward = self.reward_fn(correct[-1], total[-1], self.classes, classifier=class_net, 
            cnn_layers=len(self.cnn_state), mlp_layers=len(mlp_state))
        
        # Collate "extra info" that may be needed for logging.
        ret_dict = {'params':nasrl.reward.count_params(class_net),
            'accuracy':list(map(lambda x,y: x/y, correct, total))}
        # No need to preserve layer specifications, those are already embedded in the `state` element.

        return self._get_normalized_state(), reward, False, ret_dict

# Classification where only CNN's will be used.
class CNNClassificationEnv(JointClassificationEnv):
    def __init__(self, data_dim, cnn_conf, inner_loss=None, train_data_iter=None, validation_data_iter=None, 
    classes=10, reward_fn=nasrl.reward.Linear, adapt_steps=2):
        super(CNNClassificationEnv, self).__init__(data_dim, cnn_conf, None, inner_loss, train_data_iter, 
        validation_data_iter, classes, reward_fn=reward_fn, adapt_steps=adapt_steps)
        self.observation_space = self.cnn_observation_space
    def reset(self):
        # Only expose the state of the CNN.
        return super(CNNClassificationEnv, self).reset()[0]
    def step(self, actions):
        for action in actions:
            assert isinstance(action, ActionAddConv) or isinstance(action, ActionAddPool) or (
                isinstance(action, ActionDelete) and action.which == LayerType.CNN)
        state, reward, done, dict = super(CNNClassificationEnv, self).step(actions)
        # Only expose the state of the CNN.
        return state[0], reward, done, dict

# Classification where only MLP's will be used.
class MLPClassificationEnv(JointClassificationEnv):
    def __init__(self, data_dim, mlp_conf, inner_loss=None, train_data_iter=None, validation_data_iter=None, 
    classes=10, reward_fn=nasrl.reward.Linear, adapt_steps=2):
        super(MLPClassificationEnv, self).__init__(data_dim, None, mlp_conf, inner_loss, train_data_iter, 
        validation_data_iter, classes, reward_fn=reward_fn, adapt_steps=adapt_steps)
        self.observation_space = self.mlp_observation_space

    def reset(self):
        # Only expose the state of the MLP.
        return super(MLPClassificationEnv, self).reset()[1]
    def step(self, actions):
        for action in actions:
            assert isinstance(action, ActionAddMLP) or (
                isinstance(action, ActionDelete) and action.which == LayerType.MLP)
        state, reward, done, dict = super(MLPClassificationEnv, self).step(actions)
        # Only expose the state of the MLP.
        return state[1], reward, done, dict