"""
This file orchestrates experiment launching

"""
from .utils import seed_all

import ray
ray.rllib = None
import ray.tune
from ray.tune import register_trainable, run_experiments
import track

import argparse
import os
import pickle
import shutil
import yaml


class LaunchVar:
    """ Global Variables Are Bad """
    def __init__(self):
        self.val = None
        pass

    def set(self, val):
        """ Set the stinking value """
        self.val = val


# This will hold the arguments for this program. Set in `supply_args`.
_parser = LaunchVar()
# This will be called after all experiments have run if it is set.
# See `supply_postprocess`.
_postprocess_fn = LaunchVar()
# If set to true in `supply_postporcess`, saves the track.Project.
_save_proj = LaunchVar()


def _add_default_args(parser):
    """
    This function adds a suite of default arguments for the project that
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
                        help='if > 0, create ray host with specified # of GPUs')
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
    # Storage arguments
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
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed to supply to numpy, random, torch')


def _experiment(experiment_fn, args):
    """
    Launches the track experiment (+/- S3 backup) by calling
    `experiment_fn(args)` where args contains the parsed arguments.
    """
    # Set up the random seeds!
    seed_all(args.seed)
    # Set up track logging, locally + in S3
    track_local_dir = os.path.join(args.logroot, args.experimentname)
    if args.s3:
        track_remote_dir = os.path.join(args.s3,
                                        args.projectname,
                                        args.experimentname)
    else:
        track_remote_dir = None
    # Start the trial!
    with track.trial(track_local_dir, track_remote_dir, param_map=vars(args)):
        track.debug("Starting experiment!")
        experiment_fn(args)


def _compute_resources(args):
    cpu = 1 if args.self_host and args.cpu else 0
    gpu = 0 if args.cpu else min(args.devices_per_trial, args.self_host)
    return {'cpu': cpu, 'gpu': gpu}


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
        ray.init(num_cpus=args.self_host, num_gpus=0)
    else:
        ray.init(num_gpus=args.self_host)

    def _real_ray_exp(config, status_reporter):
        _ray_experiment(experiment_fn, args, config, status_reporter)
    register_trainable('ray_experiment', _real_ray_exp)

    with open(args.config) as f:
        config = yaml.load(f)

    resources = _compute_resources(args)
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
        run_experiments(experiment_setting,
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
    os.makedirs('raydata', exist_ok=True)  # just in case we did nothing
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
    ** This function must be called before `execute` **

    This function will call `argument_fn(parser)` where parser is the
    ArgumentParser for this python program. Certain default arguments
    related to resource allocation and logging are created first.
    Please see `_add_default_args` for defaults on these additions.

    `argument_fn(parser)`: adds user-specific arguments to the argparser.
    """
    _parser.set(argparse.ArgumentParser(description='skeletor argument parser'))
    _add_default_args(_parser.val)
    if argument_fn:
        argument_fn(_parser.val)


def supply_postprocess(postprocess_fn=None, save_proj=False):
    """
    ** This function must be called before `execute` **

    This schedules postprocessing analysis using the supplied
    `postprocess_fn(proj)` function. `proj` is a track.Project object that
    contains the results for the various experiments with
    the specified `args.experimentname`.

    Postprocessing will be called after `_experiment` or after
    `_cleanup_ray_experiments` depending on if ray was used or not.

    `save_proj`: if True, this will create a pickle file containing the
    track.Project object generated for `args.experimentname`.
    It will save to <logroot>/<experimentname>/<experimentname>.pkl
    """
    _postprocess_fn.set(postprocess_fn)
    _save_proj.set(save_proj)


def execute(experiment_fn):
    """
    Launches an experiment using the supplied `experiment_fn(args)` launcher.
    If the config is set, it will use ray to launch all experiments
    in parallel.
    """
    # Parse all arguments (default + user-supplied)
    if not _parser.val:
        supply_args()
    args = _parser.val.parse_args()
    # Launch ray if we need to.
    if args.config:
        _launch_ray_experiments(experiment_fn, args)
        _cleanup_ray_experiments(args)
    # Launch a single experiment otherwise.
    else:
        _experiment(experiment_fn, args)
    # Load resulting experiment data from Track
    local = os.path.join(args.logroot, args.experimentname)
    if args.s3:
        track_remote_dir = os.path.join(args.s3,
                                        args.projectname,
                                        args.experimentname)
    else:
        track_remote_dir = None
    proj = track.Project(local, track_remote_dir)
    # Save project to a pickle in <logroot>/<experimentname>.
    if _save_proj.val:
        proj_fname = os.path.join(args.logroot, args.experimentname,
                                  args.experimentname + '.pkl')
        try:
            with open(proj_fname, 'wb') as f:
                pickle.dump(proj, f)
        except Exception as e:
            print('swallowing pickle error: {}'.format(e))
    # Launch postprocessing code.
    if _postprocess_fn.val:
        _postprocess_fn.val(proj)
