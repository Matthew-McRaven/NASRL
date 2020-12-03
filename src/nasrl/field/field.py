# Represent a closed interval over totally ordered types.
#   https://en.wikipedia.org/wiki/Total_order
class Interval:
    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max
    # Needed to support `if # in interval:...`
    def __contains__(self, value):
        return self.min <= value <= self.max

# Represent one dimension / hyperparameter of our environment.
# That is, the number of channels, or range of sizes for a MLP layer.
# The field is aware of normalizing / denormalizing values for a NN using transform()
# However, it is on the callr to ensure that the transform is "good".
# Additionally, a field can be queried for a suggetsed initial value using init_value().
class Field:
    def __init__(self, domain:Interval, val_init_fn, transform_for_nn=None):
        self.domain = domain
        self.val_init_fn = val_init_fn
        # If no transform is provided, assume we should map the field to [0, 1]
        if transform_for_nn == None:
            from .transform import LinearTransform
            transform_for_nn = LinearTransform(domain.min, domain.max)
        self.transform = transform_for_nn

    # Generate a starting value for the field
    def init_value(self, layer_index):
        ret = self.val_init_fn(layer_index, self.domain)
        assert ret in self.domain
        return ret 