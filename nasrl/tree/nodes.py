import torch, torch.nn, torch.nn.functional, torch.distributions
from .actions import *
from .ptree import ProbabilisticLeaf

"""
Create nodes that add / remove MLPs & CNNs from a neural network.
These nodes must implement the nasrl.tree.ProbabilisticLeaf interface.
"""

# Delete a layer from a MLP.
class NodeMLPDel(ProbabilisticLeaf):
    def _sample(self, weight_dict):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['mlp_count']-1)
        layer_num = layer_select_dist.rsample()
        return ActionDelete(self, layer_num , LayerType.MLP)

    def _log_prob(self, action, weight_dict):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['mlp_count'])
        lp = layer_select_dist.log_prob(action.layer_num)
        return lp

# Delete a layer from a CNN.
class NodeConvDel(ProbabilisticLeaf):
    def _sample(self, weight_dict):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['conv_count']-1)
        return ActionDelete(self, layer_select_dist.sample(), LayerType.CNN)

    def _log_prob(self, action, weight_dict):
        layer_select_dist = torch.distributions.Uniform(0, weight_dict['conv_count']-1)
        return layer_select_dist.log_prob(action.layer_num)

# Add a layer to a MLP.
class NodeMLPAdd(ProbabilisticLeaf):
    @staticmethod
    def _get_dists( weight_dict):
        return torch.distributions.Uniform(0, weight_dict['mlp_count']), weight_dict['mlp_size_dist']

    def _sample(self, weight_dict):
        layer_select_dist, layer_size_dist = NodeMLPAdd._get_dists(weight_dict)
        layer_num, layer_size = layer_select_dist.sample(), layer_size_dist.sample()
        return ActionAddMLP(self, layer_num, layer_size)

    def _log_prob(self, action, weight_dict):
        layer_select_dist, layer_size_dist = NodeMLPAdd._get_dists(weight_dict)
        return layer_select_dist.log_prob(action.layer_num) + layer_size_dist.log_prob(action.layer_size)

# Add a convolutional layer to a CNN.
class NodeConvAdd(ProbabilisticLeaf):
    @staticmethod
    def _get_dists(weight_dict):
        dists = []
        dists.append(weight_dict['kernel_dist'])
        dists.append(weight_dict['channel_dist'])
        dists.append(weight_dict['stride_dist'])
        dists.append(weight_dict['padding_dist'])
        dists.append(weight_dict['dilation_dist'])
        return torch.distributions.Uniform(0, weight_dict['conv_count']), *dists

    def _sample(self, weight_dict):
        layer_select_dist, k_dist, c_dist, s_dist, p_dist, d_dist = NodeConvAdd._get_dists(weight_dict)
        conv_args = [x.sample() for x in (k_dist,  c_dist, s_dist, p_dist, d_dist)]
        
        return ActionAddConv(self, layer_select_dist.sample(), *conv_args)

    def _log_prob(self, action, weight_dict):
        layer_select_dist, k_dist, c_dist, s_dist, p_dist, d_dist = NodeConvAdd._get_dists(weight_dict)
        logprob = layer_select_dist.log_prob(action.layer_num)
        logprob += k_dist.log_prob(action.conv_def.kernel)
        logprob += c_dist.log_prob(action.conv_def.out_channels)
        logprob += s_dist.log_prob(action.conv_def.stride)
        logprob += p_dist.log_prob(action.conv_def.padding)
        logprob += d_dist.log_prob(action.conv_def.dilation)
        return logprob

# Add a pooling layer to a CNN.
class NodePoolAdd(ProbabilisticLeaf):
    @staticmethod
    def _get_dists(weight_dict):
        dists = []
        dists.append(weight_dict['kernel_dist'])
        dists.append(weight_dict['stride_dist'])
        dists.append(weight_dict['padding_dist'])
        dists.append(weight_dict['dilation_dist'])
        return torch.distributions.Uniform(0, weight_dict['conv_count']), *dists

    def _sample(self, weight_dict):
        pool_type = weight_dict['pool_type']
        layer_select_dist, k_dist, s_dist, p_dist, d_dist = NodeConvAdd._get_dists(weight_dict)
        pool_args = [x.sample() for x in (k_dist, s_dist, p_dist, d_dist)]
        
        return ActionAddPool(self, layer_select_dist.sample(), pool_type, *pool_args)

    def _log_prob(self, action, weight_dict):
        layer_select_dist, k_dist, s_dist, p_dist, d_dist = NodeConvAdd._get_dists(weight_dict)
        logprob = layer_select_dist.log_prob(action.layer_num)
        logprob += k_dist.log_prob(action.conv_def.kernel)
        logprob += s_dist.log_prob(action.conv_def.stride)
        logprob += p_dist.log_prob(action.conv_def.padding)
        logprob += d_dist.log_prob(action.conv_def.dilation)
        return logprob