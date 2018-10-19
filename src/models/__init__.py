""" Top-level directory for all network architectures """

# from .resnet import ResNet18


def build_model(name, model_params):
    """
    Builds a model from the specified name and parameters.
    The name must be one of the model definition functions imported above
    or defined below.
    """
    assert name in locals().keys(),\
        "%s must be a model importd/defined in models/__init__.py" % name
    return globals()[name](**model_params)
