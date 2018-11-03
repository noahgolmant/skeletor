""" Get train and test loaders for various datasets """
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skeletor.register import register_module, register_callable, build_callable

_custom_datasets = {}  # what datasets have we already registered
_custom_modules = set()  # what modules have we already imported from


def build_dataset(name, **dataset_params):
    """
    Builds dataset loader from the specified name and parameters.
    The name must be one of the callable definition functions imported above
    or defined below.

    Why might you use this? The best use-case is enabling direct
    dataset configuration through hyperparameters.
    """
    return build_callable(name, _custom_datasets,
                          globals(), **dataset_params)


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
