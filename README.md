# Skeletor

Skeletor attempts to provide a lightweight wrapper for research code with two goals: (1) make it easy to track experiment results and data for later analysis and (2) orchestrate many experiments in parallel without worrying too much. The first goal is satisfied using [track](https://github.com/richardliaw/track) for logging experiment metrics. You can get the experiment results in a nice Pandas DataFrame with it, it logs in a nice format, and it can back up to S3. The second goal is satisfied using [ray](https://github.com/ray-project/ray) to parallelize multi-gpu grid-searched experiment configurations.

I added boilerplate model, architecture, and optimizer construction functions for some basic PyTorch setups. I will try to add more as time goes on, but I don't plan on adding TensorFlow things anytime soon.

## Setup

Necessary packages are listed in `setup.py`.
Just run `pip install skeletor` to get started.

## Basic Usage

All you really have to do is supply a `supply_args(parser)` function and an `experiment_fn(parsed_args)` function. The first one takes in an `ArgumentParser` object so you can supply your own arguments to the project. The second one will take in the parsed arguments and run your experiment.

To launch a single experiment, you can do something like

`CUDA_VISIBLE_DEVICES=0 python train.py <my args> <experimentname>`


To launch experiments in parallel, you can do something like

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py <config.yaml> --self_host=4 <experimentname>`

Logs (`track` records) will be stored in `<args.logroot>/<args.experimentname>`.

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
