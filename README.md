# Skeleton

This is a skeleton repository for training different pytorch models on different datasets. It uses [track](https://github.com/richardliaw/track) for logging experiment metrics and [ray](https://github.com/ray-project/ray) to parallelize multi-gpu experiments via a grid search.

## Setup

This setup requires a linux x64 box.
Necessary packages are listed in `setup.py`.

1. Fill out the basic information required in the `.env`.
2. Fill out the setup info by running `scripts/setup.sh`
3. Create a conda environment for this project:
    a. `conda create -y -n <proj-name> python=3.5`
    b. `conda activate <proj-name>`
4. Install any machine-specific dependencies
    a. `./scripts/install.sh`
5. Install the requirements and mark this as a local package.
    a. `pip install --no-cache-dir --editable .`

## Scripts

All scripts are available in `scripts/`, and should be run from the repo root.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with approppriate flags for repo |
| `tests.sh` | runs all tests |
| `install.sh` | use python and cuda info to install packages (e.g. pytorch) |

## Running Experiments

### Basic Usage

For running a single experiment, simply specify the flags defined in the `add_train_args` function in `src/train.py`. The last argument should be a name for the experiment. For example, the following will begin to train ResNet50 on CIFAR-10:

`CUDA_VISIBLE_DEVICES=0 python src/main.py --arch resnet50 --dataset cifar10 --lr .1 resnet_cifar`

### Parallelizing Experiments

You can schedule a set of experiments by defining a YAML config that determines all the experiment settings you want to try. An example of this is found in `configs/cifar10.yaml`. This follows the `ray.tune` setup for YAML parsing. You can launch these settings in parallel like so:


`CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --self_host=4 --config configs/cifar10.yaml resnet_cifar`

## Analyzing Experiments

You can run postprocessing routines using code in `src/proc`. For example, `src/proc/track.py` will produce a pickled DataFrame containing all results for the specified experiment name. You can do visualizations by creating notebooks in `plotting/`.

