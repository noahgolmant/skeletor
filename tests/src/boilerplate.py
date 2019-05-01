""" This file tests that all the models in skeletor.models load correctly """
from skeletor.models import build_model
from skeletor.optimizers import build_optimizer
from skeletor.datasets import build_dataset

models = [
    'ResNet18',
    'ResNet50',
    'DenseNet121',
    'DPN26',
    'GoogLeNet',
    'LeNet',
    'MobileNet',
    'MobileNetV2',
    'PNASNetA',
    'PreActResNet18',
    'ResNeXt29_2x64d',
    'SENet18',
    'VGG11'
]

datasets = [
    'mnist',
    'svhn',
    'cifar10',
    'cifar100',
]


optimizers = [
    'Adam',
    'SGD'
]

dataroot = './tests/data'
batch_size = 128
eval_batch_size = 100
num_workers = 2


def test_models():
    print("Testing models")
    for name in models:
        build_model(name, num_classes=10)


def test_datasets():
    print("Testing datasets")
    for name in datasets:
        trainloader, testloader = build_dataset(name,
                                                dataroot=dataroot,
                                                batch_size=batch_size,
                                                eval_batch_size=eval_batch_size,
                                                num_workers=num_workers)


def test_optimizers():
    print("Testing optimizers")
    model = build_model('ResNet18', num_classes=10)
    for name in optimizers:
        build_optimizer(name, params=model.parameters(), lr=.1)


def test_models_for_dataset():
    for dataset_name in datasets:
        trainloader, testloader = build_dataset(dataset_name,
                                                dataroot=dataroot,
                                                batch_size=batch_size,
                                                eval_batch_size=eval_batch_size,
                                                num_workers=num_workers)
        for model_name in ['LeNet', 'ResNet18', 'ResNet50']:
                print("Testing %s with %s" % (dataset_name, model_name))
                model = build_model(model_name, dataset=dataset_name)
                x, y = next(iter(trainloader))
                model(x)


if __name__ == '__main__':
    test_models()
    test_datasets()
    test_optimizers()
    test_models_for_dataset()
