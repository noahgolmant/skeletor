""" Top-level for building an optimizer """
from torch.optim.sgd import SGD
from torch.optim.adam import Adam


def build_optimizer(name, **optimizer_params):
    """
    Builds an optimizer from the specified name and parameters.
    The name must be one of the optimizer classes imported above.
    """
    assert name in globals().keys(),\
        "%s must be a listed optimizer in optimizers/__init__.py" % name
    return globals()[name](**optimizer_params)
