# Skeletor [![Build Status](https://travis-ci.org/noahgolmant/skeletor.svg?branch=master)](https://travis-ci.org/noahgolmant/skeletor)

Skeletor attempts to provide a lightweight wrapper for research code with two goals: (1) make it easy to track experiment results and data for later analysis and (2) orchestrate many experiments in parallel without worrying too much. The first goal is satisfied using [track](https://github.com/richardliaw/track) for logging experiment metrics. You can get the experiment results in a nice Pandas DataFrame with it, it logs in a nice format, and it can back up to S3. The second goal is satisfied using [ray](https://github.com/ray-project/ray) to parallelize multi-gpu grid searches over various experiment configurations. This is an improvement over some other setups because it allows us to use a proper distributed execution framework to handle trial scheduling.

99% of the work is being done by track and ray.

I added boilerplate model, architecture, and optimizer construction functions for some basic PyTorch setups. I will try to add more as time goes on, but I don't plan on adding TensorFlow things anytime soon.

## Setup

Necessary packages are listed in `setup.py`.
Just run `pip install skeletor-ml` to get started.

## Basic Usage

All you really have to do is supply a `supply_args(parser)` function and an `experiment_fn(parsed_args)` function. The first one takes in an `ArgumentParser` object so you can supply your own arguments to the project. The second one will take in the parsed arguments and run your experiment.

You can use `track` to log statistics during training. A basic example `train.py` might look like:

```
import skeletor
from skeletor.models import build_model
from skeletor.optimizers import build_optimizer
import track

def add_args(parser):
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--lr', default=0.1, type=float)

def train(epoch):
    ...
    return avg_train_loss

def test(epoch):
    ...
    return avg_test_loss

def experiment(args):
    model = build_model(args.arch, num_classes=10)
    opt = build_optimizer('SGD', lr=args.lr)
    for epoch in range(200):
        track.debug("Starting epoch %d" % epoch)
        train_loss = train(epoch)
        test_loss = test(epoch)
        track.metric(iteration=epoch,
                     train_loss=train_loss,
                     test_loss=test_loss)

skeletor.supply_args(add_args)
skeletor.execute(experiment)
```

To launch a single experiment, you can do something like

`CUDA_VISIBLE_DEVICES=0 python train.py --arch resnet50 --lr .1 resnet_cifar`


The same code can be used to launch several experiments in parallel. Suppose I have a config called `config.yaml` that looks like:

```
arch: resnet50
lr:
  grid_search: [.001, .01, .1, 1.0]
```

I can test out all of these learning rates at the same time by running:

`CUDA_VISIBLE_DEVICES=0,1 python train.py config.yaml --self_host=2 resnet_cifar`

Ray will handle scheduling all four jobs on the two devices using a queue.

Logs (`track` records) will be stored in `<args.logroot>/<args.experimentname>`.
See the `track` docs for how to access these records as DataFrames.

## Examples

You can find an example of running a grid search for training a residual network on CIFAR-10 in PyTorch in `examples/train.py`.


## Getting experiment results

I added a utility in `skeletor.proc` for converting all `track` trial records for an experiment into a single Pandas DataFrame. It can also pickle it.

That means if I run an experiment like above called `resnet_cifar`, I can access all of the results for all the trials as a single DataFrame by calling `skeletor.proc.proj('resnet_cifar', './logs')`.

## Registering custom models, dataloaders, and optimizers

Registering custom classes allows you to construct an instance of the specified class by calling `build_model`, `build_dataset`, or `build_optimizer` with the class string name. This is useful for hyperparameter searching because you can search over these choices directly by class name.

I try to provide a simple interface for registering custom implementations with skeletor. For example, I can register a custom `Model` class by calling `skeletor.models.add_model(Model)`. This allows me to create models through `skeletor.models.build_model('Model')`. You can also register entire modules full of definitions at once. There are analogous functions `add_dataset, add_optimizer` for datasets and optimizers.

```
class MyNetwork(Module):
    ...

skeletor.models.add_model(MyNetwork)

arch_name = 'MyNetwork'
model = skeletor.models.build_model(arch_name)
```

## Help me out / Things to Do

We have active [issues](https://github.com/noahgolmant/skeletor/issues)! Feel free to suggest new improvements or add PRs to contribute.

[...](https://www.youtube.com/watch?v=g20_8-TPyTQ)
