import librl.nn.core

# Our agent only passes one argument to forward.
# That argument is really of tuple of values that can be unpacked.
# This class adapts the packed input to the two inputs expected by BilinearKernel. 
class BilinearAdapter(librl.nn.core.BilinearKernel):
    def __init__(self, *args, **kwargs):
        super(BilinearAdapter, self).__init__(*args, **kwargs)
    def forward(self, tuple_input):
        i0, i1 = tuple_input
        return super(BilinearAdapter, self).forward(i0, i1)