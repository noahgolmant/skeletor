"""
This module controls registering custom implementations of various models,
datasets, and optimizers so that it's easy to search over them using
grid searches and string configuratoins.
"""
import importlib
from types import FunctionType, ModuleType
from skeletor.error import SkeletorException
from warnings import warn


def _is_callable(cls):
    """ A model should be created through either a function or a class """
    return isinstance(cls, type) or isinstance(cls, FunctionType)


def build_callable(name, custom_callables, module_globals, **params):
    if name in custom_callables.keys():
        return custom_callables[name](**params)
    registered = name in module_globals.keys()
    if not registered:
        raise SkeletorException('%s must be registered as a callable first. '
                                'either it is in skeletor.models, or imported'
                                ' via `register_module` or `register_callable`.'
                                % name)
    return module_globals[name](**params)



def register_callable(cls, custom_callables, override=False):
    """
    Registers a custom callable with skeletor.
    After registering, you can build an instance of it
    using something like `build_model` like normal.

    If override is true, replaces the previous def of cls if it exists.
    This uses cls.__name__ to identify the class.
    """
    if not _is_callable(cls):
        raise SkeletorException('Failed to register callabe %s. '
                                'Not a function or class!' % cls)
    if cls.__name__ in custom_callables.keys() and not override:
        warn("Didn't register callable %s. Already registered w/o override."
             % cls.__name__, RuntimeWarning)
        return
    custom_callables[cls.__name__] = cls


def register_module(name, custom_callables,
                    custom_modules, override=False):
    """
    This registers a custom module with skeletor.
    After registering, you can build an instance of classes/functions defined
    within this module by using stuff like `build_model`.

    This function returns the set of classes/function that it registered.
    Right now, this registers a lot of extra things... e.g. test functions
    within these model definitions.

    The assumed file structure is either
        (a) A file that contains class/function model definitions
        (b) a module that contains several such files

    This returns a list of names of callable model definitions registered
    with skeletor.
    """
    if name in custom_modules and not override:
        return
    importlib.invalidate_caches()  # to prevent screwing with older imports
    try:
        imported = __import__(name, fromlist='*')
        _is_submodule = lambda x: isinstance(x, ModuleType)
        to_register = []
        # If there are callable model definitions at the root level, add here.
        module_callables = list(filter(_is_callable,
                                       imported.__dict__.values()))
        to_register.extend(module_callables)
        # Otherwise, iterate through all submodules, and import all callable
        # model definitions inside of these.
        for submodule in filter(_is_submodule, imported.__dict__.values()):
            submodule_callables = list(filter(_is_callable,
                                              submodule.__dict__.values()))
            to_register.extend(submodule_callables)
        # Register all available modules.
        for model in to_register:
            register_callable(model, custom_callables,
                              override=override)
        # Let the user know that we failed to find any...
        if len(to_register) == 0:
            warn("Failed to import any callable definitions for module %s" % name,
                 RuntimeWarning)
        return [registered.__name__ for registered in to_register]
    except Exception as e:
        warn("Skeletor failed to load the module %s:\n%s" % (name, e),
             RuntimeWarning)
