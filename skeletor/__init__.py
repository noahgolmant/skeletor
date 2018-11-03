"""Top-level structure for skeletor """
name = 'skeletor-ml'

from . import register
from . import proc
from . import models
from . import datasets
from . import optimizers
from .launcher import supply_args, supply_postprocess, execute
