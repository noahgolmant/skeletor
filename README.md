# Skeletor

Skeletor attempts to provide a lightweight wrapper for research code with two goals: (1) make it easy to track experiment results and data for later analysis and (2) orchestrate many experiments in parallel without worrying too much. The first goal is satisfied using [track](https://github.com/richardliaw/track) for logging experiment metrics. You can get the experiment results in a nice Pandas DataFrame with it, it logs in a nice format, and it can back up to S3. The second goal is satisfied using [ray](https://github.com/ray-project/ray) to parallelize multi-gpu grid searches over various experiment configurations. This is an improvement over some other setups because it allows us to use a proper distributed execution framework to handle trial scheduling.

99% of the work is being done by track and ray.

I added boilerplate model, architecture, and optimizer construction functions for some basic PyTorch setups. I will try to add more as time goes on, but I don't plan on adding TensorFlow things anytime soon.

## Setup

Necessary packages are listed in `setup.py`.
Just run `pip install skeletor-ml` to get started.

## Basic Usage

All you really have to do is supply a `supply_args(parser)` function and an `experiment_fn(parsed_args)` function. The first one takes in an `ArgumentParser` object so you can supply your own arguments to the project. The second one will take in the parsed arguments and run your experiment.

A basic example `train.py` might look like:

```
import skeletor
from skeletor.models import build_model
from skeletor.optimizers import build_optimizer

def add_args(parser):
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--lr', default=0.1, type=float)

def train(epoch):
    ...

def test(epoch):
    ...

def experiment(args):
    model = build_model(args.arch, num_classes=10)
    opt = build_optimizer('SGD', lr=args.lr)
    for epoch in range(200):
        train(epoch)
        test(epoch)

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

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config.yaml --self_host=4 resnet_cifar`

If I have more than 4 configurations, ray will handle job scheduling from the queue.

Logs (`track` records) will be stored in `<args.logroot>/<args.experimentname>`.
See the `track` docs for how to access these records as DataFrames.

## Examples

You can find an example of running a grid search for training a residual network on CIFAR-10 in PyTorch in `examples/train.py`.


## Getting experiment results

I added a utility in `skeletor.proc` for converting all `track` trial records for an experiment into a single Pandas DataFrame. It can also pickle it.

## Help me out

I tried to erase boilerplate by adding basic experiment utilities as well as various models and dataloaders. I haven't added much yet. Feel free to port over other architectures and datasets into the repo via PRs.

## Things to do

Add capability to register custom models, dataset loaders, and optimizers with the `build_model`, `build_dataset`, and `build_optimizer` functions.

Sometimes `track` doesn't install correctly from the `setup.py`. If this happens, just run `pip install --upgrade git+https://github.com/richardliaw/track.git@master#egg=track` first.
`



[...](https://www.youtube.com/watch?v=g20_8-TPyTQ)
