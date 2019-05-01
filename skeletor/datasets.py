""" Get train and test laaders for various datasets """
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skeletor.register import register_module, register_callable, build_callable
from skeletor.utils import Params

_custom_datasets = {}  # what datasets have we already registered
_custom_modules = set()  # what modules have we already imported from


# DATASET PARAMS
# These provide auxiliary information to other modules that need it
# e.g. when the convnet input channels depends on the image dimensions
_DATASET_PARAMS = {
    'mnist': Params(num_classes=10, in_channels=1, width=28, height=28),
    'svhn': Params(num_classes=10, in_channels=3, width=32, height=32),
    'cifar10': Params(num_classes=10, in_channels=3, width=32, height=32),
    'cifar100': Params(num_classes=100, in_channels=3, width=32, height=32)
}


def params(dataset_name):
    return _DATASET_PARAMS[dataset_name].args


def build_dataset(name, **extra_params):
    """
    Builds dataset loader from the specified name and parameters.
    The name must be one of the callable definition functions imported above
    or defined below.

    Why might you use this? The best use-case is enabling direct
    dataset configuration through hyperparameters.
    """
    return build_callable(name, _custom_datasets,
                          globals(), **extra_params)


def add_dataset(cls, override=True):
    register_callable(cls, _custom_datasets, override=override)


def add_module(name, override=True):
    register_module(name, _custom_datasets, _custom_modules,
                    override=override)


def num_classes(name):
    """ Gets the number of classes in specified dataset to create models """
    if name == 'cifar10':
        return 10
    elif name == 'cifar100':
        return 10
    return 0


def cifar10(**kwargs):
    """ Loads the cifar10 dataset train and test loaders """
    return _cifar(True, **kwargs)


def cifar100(**kwargs):
    """ Loads the cifar100 dataset train and test loaders """
    return _cifar(False, **kwargs)


def _cifar(is_cifar10, dataroot, batch_size, eval_batch_size,
           num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if is_cifar10:
        dataloader = datasets.CIFAR10
    else:
        dataloader = datasets.CIFAR100

    datadir = os.path.join(dataroot, 'cifar10' if is_cifar10 else 'cifar100')
    trainset = dataloader(datadir, train=True, download=True,
                          transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)

    testset = dataloader(datadir, train=False, download=False,
                         transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=eval_batch_size,
                                 shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def svhn(dataroot, batch_size, eval_batch_size, num_workers=2):
    """" train, test, num_classes for svhn house digits classification """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    trainset = datasets.SVHN(root=dataroot, split='train',
                             download=True, transform=transform)
    testset = datasets.SVHN(root=dataroot, split='test',
                            download=True, transform=transform)

    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    testloader = data.DataLoader(testset,
                                 batch_size=eval_batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    return trainloader, testloader


def mnist(dataroot, batch_size, eval_batch_size, num_workers=2):
    """returns train, test, num_classes for MNIST"""
    # per https://github.com/pytorch/examples/blob/master/mnist
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))])
    trainset = datasets.MNIST(dataroot, train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(dataroot, train=False,
                             download=True, transform=transform)

    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    testloader = data.DataLoader(testset,
                                 batch_size=eval_batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    return trainloader, testloader
