
import librl.nn.core.cnn
from numpy import lib

from .field import Field

# Describe the range of values that MLP's neurons can take on.
class MLPConf:
    def __init__(self, n_layers, mlp, init_layer_count_fn=lambda max_layers:1):
        assert n_layers>=0
        assert isinstance(mlp, Field) or (mlp is None and n_layers==0)
        self.n_layers = n_layers
        self.mlp = mlp
        self._init_layer_count = init_layer_count_fn
        
    def init_layer_count(self):
        return self._init_layer_count(self.n_layers)

# Choose sensible ranges for channels, kernel size, stride,
# padding, and dilation.
class CNNConf:
    def __init__(self, n_layers, channel, kernel, stride, padding, dilation, 
        init_layer_count_fn=lambda max_layers:0, init_layer_type_fn=lambda layer_idx:'conv'):

        assert n_layers>=0

        def helper(the_field):
            return isinstance(the_field, Field) or (the_field is None and n_layers==0)

        assert helper(channel) and helper(kernel) and helper(stride)
        assert helper(padding) and helper(dilation)

        self.n_layers = n_layers
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._init_layer_count = init_layer_count_fn
        self._init_layer_type = init_layer_type_fn

    # Query how many CNN layers to start with?
    def init_layer_count(self):
        return self._init_layer_count(self.n_layers)

    # Create a convolutional layer definition by querrying initial values from each field.
    def init_layer_value(self, layer_index) -> librl.nn.core.cnn.conpool_core:
        type = self._init_layer_type(layer_index)
        c = self.channel.init_value(layer_index)
        k = self.kernel.init_value(layer_index)
        s = self.stride.init_value(layer_index)
        p = self.padding.init_value(layer_index)

        # Force dilation to be sane.
        if type == 'avg': d = 1
        else: d= self.dilation.init_value(layer_index)

        if type == 'conv': cnn_def = librl.nn.core.cnn.conv_def(k, c, s, p, d)
        elif type == 'avg' or type == 'max': cnn_def = librl.nn.core.cnn.pool_def(k, s, p, d, pool_type=type)
        else: raise NotImplementedError(f"I don't understand a conolutional layer of type {type}.")
        return cnn_def