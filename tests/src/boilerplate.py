""" This file tests that all the models in skeletor.models load correctly """
from skeletor.models import build_model
from skeletor.optimizers import build_optimizer
from skeletor.dataset import build_dataset

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
    'cifar10',
    'cifar100'
]

optimizers = [
    'Adam',
    'SGD'
]


def test_models():
    print("Testing models")
    for name in models:
        _ = build_model(name, num_classes=10)


def test_datasets():
    print("Testing datasets")
    for name in datasets:
        trainloader, testloader = build_dataset(name,
                                                dataroot='./data/',
                                                batch_size=128,
                                                eval_batch_size=100,
                                                num_workers=1)


def test_optimizers():
    print("Testing optimizers")
    model = build_model('ResNet18', num_classes=10)
    for name in optimizers:
        _ = build_optimizer(name, params=model.parameters(), lr=.1)


if __name__ == '__main__':
    test_models()
    test_datasets()
    test_optimizers()
