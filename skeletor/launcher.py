"""
This file orchestrates experiment launching

"""
import argparse
import os
import shutil
import sys
import yaml

import ray
ray.rllib = None
import ray.tune
from ray.tune import register_trainable, run_experiments
import track

# This will hold the arguments for this program. Set in `supply_args`.
parser = None


def _add_default_args(parser):
    """
    This function adds a suite of default arguments for the projec that 
    should / must be specified by the program. Most of these can be left as
    they are. Only experimentname has to be specified, and if config is 
    specified, skeletor will launch a ray tune server to orchestrate the 
    experiments in parallel based on a tune YAML config.
    """
    # Experiment arguments 
    parser.add_argument('experimentname', type=str,
                        help='Name of the experiment to run')
    # Ray arguments
    parser.add_argument('--self_host', type=int, default=1,
                        help='if > 0, create ray host with specified number of GPUs')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--port', type=int, default=6379, help='ray port')
    parser.add_argument('--server_port', type=int, default=10000,
                        help='ray tune port')
    parser.add_argument('--config', default='', type=str,
                        help='If the config is specified, skeletor will use '
                        'ray tune to launch all experiment configurations '
                        'specified by the config YAML in parallel across a '
                        'multi-gpu machine.')
    parser.add_argument('--devices_per_trial', default=1, type=int,
                        help='number of gpus/cpus needed per experiment')
    parser.add_argument('--dataroot', default='./data', type=str,
                        help='local or absolute path where data is stored '
                        'or is to be downloaded')
    parser.add_argument('--s3', default='', type=str,
                        help='if set, this should be a valid s3 location in '
                        'a bucket where track logs will be backed up in..')
    parser.add_argument('--logroot', default='./logs', type=str,
                        help='each directory in the logroot will be an '
                        'experimentname. each such subdirectory contains all'
                        ' track records for the experiment.')


def _experiment(experiment_fn, args):
    """
    Launches the track experiment (+/- S3 backup) by calling
    `experiment_fn(args)` where args contains the parsed arguments.
    """
    track_local_dir = os.path.join(args.logroot, args.experimentname)
    if args.s3:
        track_remote_dir = os.path.join(args.s3,
                                        args.projectname,
                                        args.experimentname)
    else:
        track_remote_dir = None
    with track.trial(track_local_dir, track_remote_dir, param_map=vars(args)):
        track.debug("Starting experiment!")
        experiment_fn(args)


def _compute_resources(args, config):
    cpu = 1 if args.self_host and args.cpu else 0
    # TODO use batch size to fix this
    return {'cpu': cpu, 'gpu': args.devices_per_trial}


def _ray_experiment(experiment_fn, args, config, status_reporter):
    """ This is the actor that will start the track trial """
    for k, v in config.items():
        setattr(args, k, v)
    status_reporter(timesteps_total=0, done=0)
    _experiment(experiment_fn, args)
    status_reporter(timesteps_total=1, done=1)


def _launch_ray_experiments(experiment_fn, args):
    """
    Initialize ray and the ray tune server. Parse the config.
    Allocate resources per experiment.
    Schedule all the trials and start them all.
    """
    if args.cpu:
        ray.init(num_cpus=args.self_host)
    else:
        ray.init(num_gpus=args.self_host)
    register_trainable('ray_experiment', _ray_experiment)

    with open(args.config) as f:
        config = yaml.load(f)

    resources = _compute_resources(args, config)
    experiment_setting = {
        args.experimentname: {
            'run': 'ray_experiment',
            'trial_resources': resources,
            'stop': {
                'done': 1
            },
            'config': config,
            'local_dir': './raydata'
        }
    }

    try:
        run_experiments(experiment_fn,
                        args,
                        experiment_setting,
                        server_port=int(args.server_port),
                        with_server=True)
    except ray.tune.error.TuneError as e:
        print('swalling tune error: {}'.format(e))


def _cleanup_ray_experiments(args):
    """
    Takes the ray results from ./raydata and moves them into
    `<args.logroot>/<args.experimentname>` for permanent storage.
    """
    track_local_dir = os.path.join(args.logroot, args.experimentname)
    os.makedirs(track_local_dir, exist_ok=True)
    for experiment in os.listdir('raydata'):
        if experiment != args.experimentname:
            continue
        experiment_dir = os.path.join('raydata', experiment)
        for runname in os.listdir(experiment_dir):
            rundir = os.path.join(experiment_dir, runname,
                                  'logs',
                                  args.experimentname)
            for f in os.listdir(rundir):
                cur = os.path.join(rundir, f)
                new_dst = os.path.join(track_local_dir, f)
                if f != 'trials':
                    shutil.move(cur, new_dst)
                    continue
                if os.path.isdir(new_dst):
                    ray_trial_dir = os.path.join(rundir, f)
                    for trial_data in os.listdir(ray_trial_dir):
                        ray_trial_data = os.path.join(ray_trial_dir, trial_data)
                        new_trial_data = os.path.join(new_dst, trial_data)
                        shutil.move(ray_trial_data, new_trial_data)
                else:
                    shutil.move(cur, new_dst)


def supply_args(argument_fn=None):
    """
    This function will call `argument_fn(parser)` where parser is the
    ArgumentParser for this python program. Certain default arguments
    related to resource allocation and logging are created first.
    Please see `_add_default_args` for defaults on these additions.

    `argument_fn(parser)`: adds user-specific arguments to the argparser.
    """
    global parser
    parser = argparse.ArgumentParser(description='skeletor argument parser')
    _add_default_args(parser)
    if argument_fn:
        argument_fn(parser)


def execute(experiment_fn):
    """
    Launches an experiment using the supplied `experiment_fn(args)` launcher.
    If the config is set, it will use ray to launch all experiments
    in parallel.
    """
    if not parser:
        supply_args()
    args = parser.parse_args()
    if args.config:
        _launch_ray_experiments(experiment_fn, args)
        _cleanup_ray_experiments(args)
    else:
        _experiment(experiment_fn, args)
