import torch.distributions, torch.nn, torch.nn.functional

import librl.nn.core.cnn as cnn

from .ptree import *
from .nodes import *

# A decision tree that adds/deletes either MLPs or CNNs.
def FullDecisionTree():
    return ProbabalisticBranch([MLPDecisionTree(), CNNDecisionTree()], ['w_mlp', 'w_conv'])

# A decision tree that adds/deletes only MLPs.
def MLPDecisionTree():
    node_mlp_del = NodeMLPDel() # Needs 'mlp_count
    node_mlp_add = NodeMLPAdd() # Needs 'mlp_count' and 'mlp_size_dist' 
    return ProbabalisticBranch([node_mlp_del, node_mlp_add], ['w_mlp_del', 'w_mlp_add']) 

# A decision tree that adds/deletes only CNNs.
def CNNDecisionTree():
    node_conv_del = NodeConvDel() # Needs 'conv_count'
    # Needs 'conv_count', 'kernel_dist', 'channel_dist', 'stride_dist', 'padding_dist', 'dilation_dist'
    node_conv_add_conv = NodeConvAdd() 
    # Needs 'conv_count', 'kernel_dist', 'stride_dist', 'padding_dist', 'dilation_dist'
    node_conv_add_max = NodePoolAdd('max')
    node_conv_add_avg = NodePoolAdd('avg')
    node_conv_add = ProbabalisticBranch([node_conv_add_conv, node_conv_add_max, node_conv_add_avg], ['w_conv_add_conv', 'w_conv_add_max', 'w_conv_add_avg'])
    return ProbabalisticBranch([node_conv_del, node_conv_add], ['w_conv_del', 'w_conv_add'])

# Represent a policy that uses a probabilistic decision tree to generate actions.
class TreePolicy:
    def __init__(self, decision_tree, weights):
        #print(decision_tree)
        self.decision_tree = decision_tree
        self.weights = weights
    def sample(self, count):
        return self.decision_tree.sample(count, self.weights)
    def log_prob(self, actions):
        # librl expects a tensor of logprobs matching the shape of actions.
        return torch.stack([action.log_prob(self.weights) for action in actions])
