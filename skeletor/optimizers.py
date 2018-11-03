""" Top-level for building an optimizer """
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from skeletor.register import register_module, register_callable, build_callable


_custom_optimizers = {}  # what optimizers have we already registered
_custom_modules = set()  # what modules have we already imported from


def build_optimizer(name, **optimizer_params):
    """
    Builds an optimizer from the specified name and parameters.
    The name must be one of the callable definition functions imported above
    or defined below.

    Why might you use this? The best use-case is enabling direct
    optimizer configuration through hyperparameters.
    """
    return build_callable(name, _custom_optimizers,
                          globals(), **optimizer_params)


def add_optimizer(cls, override=True):
    register_callable(cls, _custom_optimizers, override=override)


def add_module(name, override=True):
    register_module(name, _custom_optimizers, _custom_modules,
                    override=override)
