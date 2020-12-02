import librl.replay
import librl.nn.core

  
def create_nn_from_def(input_dimension, conv_layers=None, linear_layers=None):
    assert not (conv_layers == None and linear_layers == None)
    module_list = []
    intermediate_dim=input_dimension
    if conv_layers:
        # Last (2) elements of input dim should be xy, 0'th should be # channels.
        module_list.append(librl.nn.core.cnn.ConvolutionalKernel(conv_layers, intermediate_dim[1:],intermediate_dim[0]))
        intermediate_dim = module_list[0].output_dimension
    if linear_layers: module_list.append(librl.nn.core.MLPKernel(intermediate_dim, linear_layers))
    return librl.nn.core.SequentialKernel(module_list)