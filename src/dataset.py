""" Get train and test loaders for various datasets """
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def build_dataset(name, **dataset_params):
    assert name in globals().keys(),\
        "%s must be a dataset in dataset.py"
    return globals()[name](**dataset_params)


def num_classes(name):
    if name == 'cifar10':
        return 10
    elif name == 'cifar100':
        return 10
    return 0


def cifar10(**kwargs):
    return _cifar(True, **kwargs)


def cifar100(**kwargs):
    return _cifar(False, **kwargs)


def _cifar(is_cifar10, dataroot, batch_size, eval_batch_size,
           num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if is_cifar10:
        dataloader = datasets.CIFAR10
    else:
        dataloader = datasets.CIFAR100

    datadir = os.path.join(dataroot, 'cifar10')
    trainset = dataloader(datadir, train=True, download=True,
                          transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)

    testset = dataloader(datadir, train=False, download=False,
                         transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=eval_batch_size,
                                 shuffle=False, num_workers=num_workers)
    return trainloader, testloader
