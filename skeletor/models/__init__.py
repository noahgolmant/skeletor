""" Top-level directory for all network architectures """
from .cv.vgg import *
from .cv.dpn import *
from .cv.lenet import *
from .cv.senet import *
from .cv.pnasnet import *
from .cv.densenet import *
from .cv.googlenet import *
from .cv.resnet import *
from .cv.resnext import *
from .cv.preact_resnet import *
from .cv.mobilenet import *
from .cv.mobilenetv2 import *

def build_model(name, **model_params):
    """
    Builds a model from the specified name and parameters.
    The name must be one of the model definition functions imported above
    or defined below.
    """
    assert name in globals().keys(),\
        "%s must be a model imported/defined in models/__init__.py" % name
    return globals()[name](**model_params)
