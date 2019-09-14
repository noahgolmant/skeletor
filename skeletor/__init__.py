"""Top-level structure for skeletor """
from . import proc
from .launcher import supply_args, supply_postprocess, execute
__all__ = ['proc', 'supply_args', 'supply_postprocess', 'execute']

name = 'skeletor-ml'
