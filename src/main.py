""" Top-level thing to run training procedures """
from train import do_training, add_train_args

import argparse
import os
import sys
import shutil
import yaml
from dotenv import load_dotenv, find_dotenv

import ray
ray.rllib = None
import ray.tune
from ray.tune import register_trainable, run_experiments
import track


def _add_generic_args(parser):
    # Generic arguments 
    parser.add_argument('experimentname', type=str,
                        help='Name of the experiment to run')
    # Ray arguments
    parser.add_argument('--self_host', type=int, default=1,
                        help='if > 0, create ray host with specified number of GPUs')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--port', type=int, default=6379, help='ray port')
    parser.add_argument('--server_port', type=int, default=10000,
                        help='ray tune port')
    parser.add_argument('--config', default='', type=str)

    # Hack: add the dotenv ones and we'll plug them in later
    parser.add_argument('--projectname', default='', type=str)
    parser.add_argument('--dataroot', default='', type=str)
    parser.add_argument('--remote', default='', type=str)
    parser.add_argument('--logroot', default='', type=str)


def _append_dotenv_args(args):
    # Append the environment information if we want it
    # It's all stored in the path now
    args.projectname = os.environ.get('projectname')
    args.dataroot = os.environ.get('dataroot')
    args.remote = os.environ.get('remote')
    args.logroot = os.environ.get('logroot')


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
parser = argparse.ArgumentParser(description='parser for %s'
                                 % os.environ.get('projectname'))
_add_generic_args(parser)
add_train_args(parser)
args = parser.parse_args()
_append_dotenv_args(args)


def experiment(args):
    track_local_dir = os.path.join(args.logroot, args.experimentname)
    if args.remote:
        track_remote_dir = os.path.join(args.remote,
                                        args.projectname,
                                        args.experimentname)
    else:
        track_remote_dir = None
    with track.trial(track_local_dir, track_remote_dir, param_map=vars(args)):
        track.debug("Starting trial")
        do_training(args)


def compute_resources(args, config):
    cpu = 1 if args.self_host and args.cpu else 0
    # TODO use batch size to fix this
    return {'cpu': cpu, 'gpu': 1 - cpu}


def ray_experiment(config, status_reporter):
    global args
    for k, v in config.items():
        setattr(args, k, v)
    status_reporter(timesteps_total=0, done=0)
    experiment(args)
    status_reporter(timesteps_total=1, done=1)


def launch_ray_experiments(args):
    if args.cpu:
        ray.init(num_cpus=args.self_host)
    else:
        ray.init(num_gpus=args.self_host)
    register_trainable('ray_experiment', ray_experiment)

    with open(args.config) as f:
        config = yaml.load(f)

    resources = compute_resources(args, config)
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


def cleanup_ray_experiments(args):
    track_local_dir = os.path.join(args.logroot, args.experimentname)
    for experiment in os.listdir('raydata'):
        if experiment != args.experimentname:
            continue
        experiment_dir = os.path.join('raydata', experiment)
        for runname in os.listdir(experiment_dir):
            rundir = os.path.join(experiment_dir, runname)
            for f in os.listdir(rundir):
                cur = os.path.join(rundir, f)
                new_dst = os.path.join(track_local_dir, f)
                if f != 'trials':
                    shutil.move(cur, new_dst)
                    continue
                if os.path.isdir(new_dst):
                    ray_trial_dir = os.path.join(rundir, f)
                    for trial_data in os.path.listdir(ray_trial_dir):
                        ray_trial_data = os.path.join(ray_trial_dir, trial_data)
                        new_trial_data = os.path.join(new_dst, trial_data)
                        shutil.move(ray_trial_data, new_trial_data)
                else:
                    shutil.move(cur, new_dst)


if __name__ == '__main__':
    if args.config:
        print(">>> Using ray to launch experiments from config")
        launch_ray_experiments(args)
        cleanup_ray_experiments(args)
    else:
        experiment(args)
