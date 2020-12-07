import math

import torch, torch.nn, torch.nn.functional, torch.distributions
from .actions import *
from .ptree import ProbabilisticLeaf

"""
Create nodes that add / remove MLPs & CNNs from a neural network.
These nodes must implement the nasrl.tree.ProbabilisticLeaf interface.
"""

# Delete a layer from a MLP.
class NodeMLPDel(ProbabilisticLeaf):
    def _sample(self, weight_dict, device):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['mlp_count']-1)
        layer_num = layer_select_dist.rsample()
        return ActionDelete(self, device, layer_num , LayerType.MLP)

    def _log_prob(self, action, weight_dict, device):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['mlp_count'])
        lp = layer_select_dist.log_prob(action.layer_num)
        assert not torch.isnan(lp).any() and torch.isfinite(lp).all()
        return lp

# Delete a layer from a CNN.
class NodeConvDel(ProbabilisticLeaf):
    def _sample(self, weight_dict, device):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['conv_count']-1)
        return ActionDelete(self, device, layer_select_dist.sample(), LayerType.CNN)

    def _log_prob(self, action, weight_dict, device):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['conv_count']-1)
        lp = layer_select_dist.log_prob(action.layer_num)
        print(lp)
        assert not torch.isnan(lp).any()
        return lp
# Add a layer to a MLP.
class NodeMLPAdd(ProbabilisticLeaf):
    @staticmethod
    def _get_dists(weight_dict):
        return torch.distributions.Uniform(0, weight_dict['mlp_count']), weight_dict['mlp_size_dist']

    def _sample(self, weight_dict, device):
        layer_select_dist, layer_size_dist = NodeMLPAdd._get_dists(weight_dict)
        layer_num, layer_size = layer_select_dist.sample(), layer_size_dist.sample()
        return ActionAddMLP(self, device, layer_num, layer_size)

    def _log_prob(self, action, weight_dict, device):
        layer_select_dist, layer_size_dist = NodeMLPAdd._get_dists(weight_dict)
        ldist, lsize = layer_select_dist.log_prob(action.layer_num), layer_size_dist.log_prob(action.layer_size)
        assert not torch.isnan(ldist).any()
        assert torch.isfinite(ldist).all()
        assert not torch.isnan(lsize).any()
        assert torch.isfinite(lsize).all()
        return ldist + lsize

# Add a convolutional layer to a CNN.
class NodeConvAdd(ProbabilisticLeaf):
    @staticmethod
    def _get_dists(weight_dict):
        dists = []
        dists.append(weight_dict['channel_dist'])
        dists.append(weight_dict['kernel_dist'])
        dists.append(weight_dict['stride_dist'])
        dists.append(weight_dict['padding_dist'])
        dists.append(weight_dict['dilation_dist'])
        return torch.distributions.Uniform(0, weight_dict['conv_count']), *dists

    def _sample(self, weight_dict, device):
        layer_select_dist, c_dist, k_dist, s_dist, p_dist, d_dist = NodeConvAdd._get_dists(weight_dict)
        conv_args = [x.sample() for x in (c_dist, k_dist, s_dist, p_dist, d_dist)]
        # Prevent runtime error when padding_size > .5*kernel_size.
        # TODO: Danger Will Robinson! Our space isn't linear.
        # Nor are we assured that kernel size and padding are on the same scale.
        # You probably got here by grep'ing for 
        #   `RuntimeError: pad should be smaller than half of kernel size, but got padW = ...`
        # A suggested refactor would be to store the padding, kernel transformations in __init__
        # backward() the numbers, clamp, and forward() the result.
        # TODO: NodePoolAdd._sample
        conv_args[3] = torch.clamp(conv_args[3], max=math.floor(conv_args[0]/2))
        
        return ActionAddConv(self, device, layer_select_dist.sample(), *conv_args)

    def _log_prob(self, action, weight_dict, device):
        layer_select_dist, c_dist, k_dist, s_dist, p_dist, d_dist = NodeConvAdd._get_dists(weight_dict)
        logprob = [layer_select_dist.log_prob(action.layer_num)]
        logprob.append(c_dist.log_prob(action.channel))
        logprob.append(k_dist.log_prob(action.kernel))
        logprob.append(s_dist.log_prob(action.stride))
        logprob.append(p_dist.log_prob(action.padding))
        logprob.append(d_dist.log_prob(action.dilation))
        slp = sum(logprob)

        assert not torch.isnan(slp).any()
        assert torch.isfinite(slp).all()

        return slp

# Add a pooling layer to a CNN.
class NodePoolAdd(ProbabilisticLeaf):
    def __init__(self, pool_type):
        self.pool_type = pool_type
    @staticmethod
    def _get_dists(weight_dict):
        dists = []
        dists.append(weight_dict['kernel_dist'])
        dists.append(weight_dict['stride_dist'])
        dists.append(weight_dict['padding_dist'])
        dists.append(weight_dict['dilation_dist'])
        return torch.distributions.Uniform(0, weight_dict['conv_count']), *dists

    def _sample(self, weight_dict, device):
        pool_type = self.pool_type
        layer_select_dist, k_dist, s_dist, p_dist, d_dist = NodePoolAdd._get_dists(weight_dict)
        pool_args = [x.sample() for x in (k_dist, s_dist, p_dist, d_dist)]
        # TODO: See warning on NodeConvAdd._sample for why this doesn't work.
        # Prevent runtime error when padding_size > .5*kernel_size.
        pool_args[2] = torch.clamp(pool_args[2], max=math.floor(pool_args[0]/2))
        return ActionAddPool(self, device, layer_select_dist.sample(), pool_type, *pool_args)

    def _log_prob(self, action, weight_dict, device):
        layer_select_dist, k_dist, s_dist, p_dist, d_dist = NodePoolAdd._get_dists(weight_dict)
        logprob = [layer_select_dist.log_prob(action.layer_num)]
        logprob.append(k_dist.log_prob(action.kernel))
        logprob.append(s_dist.log_prob(action.stride))
        logprob.append(p_dist.log_prob(action.padding))
        if not self.pool_type == 'avg': 
            logprob.append(d_dist.log_prob(action.dilation))
        slp = sum(logprob)
        assert not torch.isnan(slp).any() and torch.isfinite(slp).all()
        return slp