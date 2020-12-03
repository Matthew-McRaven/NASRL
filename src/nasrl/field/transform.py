import functools

import scipy.special

# Linearly map points in interval [old_min, old_max] to points in
# [new_min, new_max] and back. Types ought to be real, or inverse
# mapping will likely break.
class LinearTransform:
    def __init__(self, old_min, old_max, new_min=0., new_max=1.):
        self.old_min = old_min
        self.old_max = old_max
        self.new_min = new_min
        self.new_max = new_max

    def forward(self, value):
        mul = (self.new_max - self.new_min) / (self.old_max - self.old_min)
        return self.new_min + mul*(value - self.old_min)

    def backward(self, value):
        mul =  (self.old_max - self.old_min) / (self.new_max - self.new_min)
        return self.old_min + mul*(value - self.new_min)

# Remap values in [0,1] to other values in [0,1] using the beta distribution.
# Helps make "lopsided" distributions we can use to influence numeric shape.
class BetaTransform:
    def __init__(self, alpha=2., beta=5.):
        self.alpha = alpha
        self.beta = beta

    def forward(self, value):
        # I_x has domain of [0., 1.]
        assert 0.0 <= value <= 1.0
        return scipy.special.betainc(self.alpha, self.beta, value)

    def backward(self, value):
        # I_{x}^{-1} has domain of [0., 1.]
        assert 0.0 <= value <= 1.0
        return scipy.special.betaincinv(self.alpha, self.beta, value)

# Allow for function composition.
class SequentialTransform:
    def __init__(self, transforms):
        self.transfoms = transforms
    def forward(self, value):
        return functools.reduce(lambda v, tx: tx.forward(v), self.transfoms, value)
    def backward(self, value):
        return functools.reduce(lambda v, tx: tx.backward(v), reversed(self.transfoms), value)