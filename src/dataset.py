

def build_dataset(name, dataset_params):
    assert name in globals().keys(),\
        "%s must be a dataset in dataset.py"
    return globals()[name](**dataset_params)


def num_classes(name):
    if name == 'cifar10':
        return 10
    return 0


def cifar10():
    raise NotImplementedError()
