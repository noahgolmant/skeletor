""" Top-level directory for all network architectures """

from .cv.resnet import resnet50, resnet20
from .cv.vgg import vgg16_bn as vgg_16

def build_model(name, **model_params):
    """
    Builds a model from the specified name and parameters.
    The name must be one of the model definition functions imported above
    or defined below.
    """
    assert name in globals().keys(),\
        "%s must be a model imported/defined in models/__init__.py" % name
    return globals()[name](**model_params)

