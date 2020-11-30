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

from ..task import *
from .actions import *

# Environment for creating NN's that classify datasets very well.
class MLPClassificationEnv(gym.Env):
    def __init__(self, data_dim, linear_count, inner_loss=None, train_data_iter=None, validation_data_iter=None, labels=10):
        assert not isinstance(train_data_iter, torch.utils.data.DataLoader) # type: ignore
        assert not isinstance(validation_data_iter, torch.utils.data.DataLoader) # type: ignore

        # Limit the size of a neural network.
        # TODO: Actually respect this limit.
        self.observation_space = gym.spaces.Box(0, 400, (linear_count,), dtype=np.int16)
        # TODO: Figure out a sane representation of our action space.
        self.labels = labels
        self.data_dim = data_dim

        self.inner_loss = inner_loss
        self.train_data_iter = train_data_iter
        self.validation_data_iter = validation_data_iter

    # TODO: Generate an observation that isn't just a random array.
    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, actions):
        # Apply each modification in our action array.
        for action in actions:
            # Add neurons by performing a rotate right from the insertion point, and setting insertion point.
            if isinstance(action, ActionAddMLP):
                    self.state[action.layer_num:] = np.roll(self.state[action.layer_num:], 1)
                    self.state[action.layer_num] = action.layer_size
            elif isinstance(action, ActionDelete) and action.which == LayerType.MLP:
                # Remove neurons by performing a rotate left from the insertion point, and clearing the last element.
                # TODO: Prevent deleting the last layer, and penalize NN heavily for deleting a last layer.
                if np.count_nonzero(self.state) == 1: assert 0
                else:
                    self.state[action.layer_num:] = np.roll(self.state[action.layer_num:], -1)
                    self.state[-1] = 0
            # TODO: Add a no-op action.
            else: raise NotImplementedError("Not a valid action")

        print(self.state)

        # Convert array to list, ingoring any empty layers
        as_list = [x for x in self.state if x>0]

        # Create a classification network using our current state.
        class_kernel = create_nn_from_def(self.data_dim, None, as_list)
        class_net = librl.nn.classifier.Classifier(class_kernel, self.labels)
        class_net.train()

        # Create and run a classification task.
        t, v = self.train_data_iter, self.validation_data_iter
        cel = torch.nn.CrossEntropyLoss()
        inner_task = librl.task.classification.ClassificationTask(classifier=class_net, criterion=cel, train_data_iter=t, validation_data_iter=v)
        correct, total = test_gen_classifier(inner_task)
        # Reward is % accuracy on validation set.
        return self.state, 100 * correct / total, False, {}
