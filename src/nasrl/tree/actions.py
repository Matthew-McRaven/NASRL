import enum

import nasrl.field
"""
Wrap information needed to add/delete a layer inside of a class.
This will allow type checking in our environment when creating new NNs.
"""
# Actions
class LayerType(enum.Enum):
    MLP = enum.auto()
    CNN = enum.auto()
    
class Action:
    def __init__(self, parent):
        self.parent = parent
    def log_prob(self, weight_dict):
        return self.parent.log_prob(self, weight_dict)

class ActionDelete(Action):
    def __init__(self, parent, layer_num, which):
        super(ActionDelete, self).__init__(parent)
        assert isinstance(which, LayerType)
        self.layer_num = int(layer_num)
        self.which = which
    def __repr__(self):
        return f"<delete {self.which} @ {self.layer_num}>"

class ActionAddMLP(Action):
    def __init__(self, parent, layer_num, layer_size):
        super(ActionAddMLP, self).__init__(parent)
        self.layer_num = int(layer_num)
        self.layer_size = layer_size
    def __repr__(self):
        return f"<add mlp @ {self.layer_num} w/ {self.layer_size}>"

class ActionAddConv(Action):
    def __init__(self, parent, layer_num, channel, kernel, stride, padding, dilation):
        super(ActionAddConv, self).__init__(parent)
        self.layer_num = int(layer_num)
        self.channel, self.kernel, self.stride, self.padding, self.dilation = channel, kernel, stride, padding, dilation

class ActionAddPool(Action):
    def __init__(self, parent, layer_num, pool_type, kernel, stride, padding, dilation):
        super(ActionAddPool, self).__init__(parent)
        self.layer_num = int(layer_num)
        self.pool_type,self.kernel, self.stride, self.padding, self.dilation = pool_type, kernel, stride, padding, dilation