import track

from .dataset import build_dataset, num_classes
from .models import build_model


def _get_dataset(args):
    name = args['dataset']
    dataset_params = {}
    return build_dataset(name, dataset_params)


def _get_model(args):
    name = args['arch']
    model_params = {'num_classes': num_classes(args['dataset'])}
    return build_model(name, model_params)


def _train(epoch):
    raise NotImplementedError()


def _test(epoch):
    raise NotImplementedError()


def do_training(args):
    dataset = _get_dataset(args)
    model = _get_model(args)
    raise NotImplementedError()
    # For example... 
    epochs = args['epochs']
    for epoch in range(epochs):
        track.debug("Starting epoch %d" % epoch)
        _train(epoch)
        _test(epoch)
