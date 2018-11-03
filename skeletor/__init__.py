"""Top-level structure for skeletor """
from . import proc
from . import models
from . import datasets
from . import optimizers
from .launcher import supply_args, supply_postprocess, execute

__all__ = ['proc', 'supply_args', 'supply_postprocess', 'execute', 'models',
           'datasets', 'optimizers']

name = 'skeletor-ml'
