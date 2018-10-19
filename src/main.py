""" Top-level thing to run training procedures """
from .args import get_parser, parse
from .train import do_training

import os
import sys
import shutil
import yaml

import ray
ray.rllib = None
import ray.tune
from ray.tune import register_trainable, run_experiments
import track


# Argument parsing
parser = get_parser()
parser.add_argument('--log-interval', default=10,
                    help='frequency (in iters) of logging')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
args = parse()


def experiment(args):
    track_local_dir = os.path.join(args['logroot'], args['experimentname'])
    track_remote_dir = os.path.join(args['remote'],
                                    args['projectname'],
                                    args['experimentname'])
    with track.trial(track_local_dir, track_remote_dir, param_map=args):
        track.debug("Starting trial")
        do_training(args)


def compute_resources(args, config):
    cpu = 1 if args['self_host'] and args['cpu'] else 0
    # TODO use batch size to fix this
    return {'cpu': cpu, 'gpu': 1 - cpu}


def ray_experiment(config, status_reporter):
    # TODO CONVERT TO ARGS DICT HERE
    args = config
    status_reporter(timesteps_total=0, done=0)
    experiment(args)
    status_reporter(timesteps_total=1, done=1)


def launch_ray_experiments(args):
    if args['self_host']:
        if args['cpu']:
            ray.init(num_cpus=args['self_host'])
        else:
            ray.init(num_gpus=args['self_host'])
    else:
        ip = ray.services.get_node_ip_address()
        ray.init(redis_address=(ip + ':' + args['port']))
    register_trainable('ray_experiment', ray_experiment)

    with open(args['config']) as f:
        config = yaml.load(f)

    resources = compute_resources(args, config)
    experiment_setting = {
        args['experimentname']: {
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
                        server_port=int(args['server_port']),
                        with_server=True)
    except ray.tune.error.TuneError as e:
        print('swalling tune error: {}'.format(e), file=sys.stderr)


def cleanup_ray_experiments(args):
    track_local_dir = os.path.join(args['logroot'], args['experimentname'])
    for experiment in os.listdir('raydata'):
        if experiment != args['experimentname']:
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
    if args['onfig']:
        print(">>> Using ray to launch experiments from config")
        launch_ray_experiments(args)
        cleanup_ray_experiments(args)
    else:
        experiment(args)
