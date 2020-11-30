import functools

import more_itertools
import torch
import torch.nn as nn
import torch.distributions, torch.nn.init
import torch.optim

from .policy import MLPDecisionTree, TreePolicy

# Actor that generates weightings for nasrl.tree.MLPDecisionTree
class MLPTreeActor(nn.Module):
    def __init__(self, neural_module,  observation_space, output_dimension=(10,)):
        super(MLPTreeActor, self).__init__()
        self.observation_space = observation_space
        self.decision_tree = MLPDecisionTree()

        self.input_dimension = list(more_itertools.always_iterable(neural_module.output_dimension))
        self.__input_size = functools.reduce(lambda x,y: x*y, self.input_dimension, 1)
        self.neural_module = neural_module
        self.output_dimension = output_dimension
        self.__output_size = functools.reduce(lambda x,y: x*y, self.output_dimension, 1)
        self.__output_size = torch.tensor(self.__output_size, requires_grad=True, dtype=torch.float)

        # Our output layers are used as the seed for some set of random number generators.
        # These random number generators are used to generate edge pairs.
        self.layer_size = nn.Linear(self.__input_size, 1)
        self.w_mlp_del = nn.Linear(self.__input_size, 1)
        self.w_mlp_add = nn.Linear(self.__input_size, 1)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def recurrent(self):
        return self.neural_module.recurrent()
        
    def save_hidden(self):
        assert self.recurrent()
        return self.neural_module.save_hidden()

    def restore_hidden(self, state=None):
        assert self.recurrent()
        self.neural_module.restore_hidden(state)

    def forward(self, input):
        output = self.neural_module(input).view(-1, self.__input_size)

        # TODO: Figure out how to make inputs/outputs sane sizes.
        weight_dict = {}
        w_mlp_del, w_mlp_add = self.w_mlp_del(output), self.w_mlp_add(output)
        weight_dict['mlp_count'] = self.__output_size
        weight_dict['w_mlp_del'] = w_mlp_del
        weight_dict['w_mlp_add'] = w_mlp_add
        # TODO: Make min/max number of neurons a parameter of the NN.
        # Probably pull from observation space.
        base = 25
        max = torch.clamp((self.layer_size(output)+1)**2 + base, 0, 4000)
        weight_dict['mlp_size_dist'] = torch.distributions.Uniform(base, max)
        print(weight_dict)
        # Encapsulate our poliy in an object so downstream classes don't
        # need to know what kind of distribution to re-create.
        policy = TreePolicy(self.decision_tree, weight_dict)

        actions = policy.sample((1,))
        # Each actions is drawn independtly of others, so joint prob
        # is all of them multiplied together. However, since we have logprobs,
        # we need to sum instead.
        log_prob = sum(policy.log_prob(actions)) # type: ignore

        return actions, log_prob, policy
