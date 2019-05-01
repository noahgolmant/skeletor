""" This file tests that custom model/opt/dataset registration is alright """
import skeletor.models
import skeletor.optimizers
import skeletor.datasets


class Test:
    pass


def test_callable():
    skeletor.models.add_model(Test)
    t = skeletor.models.build_model('Test')
    skeletor.optimizers.add_optimizer(Test)
    t = skeletor.optimizers.build_optimizer('Test')
    skeletor.datasets.add_dataset(Test)
    t = skeletor.datasets.build_dataset('Test')


def test_modules():
    skeletor.models.add_module('module')
    t = skeletor.models.build_model('Test2')
    skeletor.optimizers.add_module('module')
    t = skeletor.optimizers.build_optimizer('Test2')
    skeletor.datasets.add_module('module')
    t = skeletor.datasets.build_dataset('Test2')


if __name__ == '__main__':
    test_callable()
    test_modules()
