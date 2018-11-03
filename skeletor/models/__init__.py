"""
Top-level directory for all network architectures.
You can register model definitions with skeletor here.

Why should you even use this instead of just building a model yourself?

- `build_model` lets you use prepackaged models from skeletor.models by default
- You can proceed to select architectures via arguments (good for grid search!)
"""
from skeletor.register import register_module, register_callable, build_callable


_custom_models = {}  # what models have we already registered
_custom_modules = set()  # what modules have we already imported from


def build_model(name, **model_params):
    """
    Builds a model from the specified name and parameters.
    The name must be one of the model definition functions imported above
    or defined below.

    Why might you use this? The best use-case is enabling direct
    model configuration through hyperparameters.
    """
    return build_callable(name, _custom_models, globals(), **model_params)


def add_model(cls, override=True):
    register_callable(cls, _custom_models, override=override)


def add_module(name, override=True):
    register_module(name, _custom_models, _custom_modules, override=override)


# Some default model definitions
add_module('skeletor.models.cv')


