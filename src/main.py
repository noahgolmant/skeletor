""" Top-level thing to run training procedures """
import os
import track

from .args import get_parser, parse
from .train import do_training

### Argument parsing
parser = get_parser()
parser.add_argument('--log-interval', default=10,
                    help='frequency (in iters) of logging')
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
args = parse()


def experiment(args):
    track_local_dir = os.path.join(args['logroot'], args['experimentname'])
    track_remote_dir = os.path.join(args['remote'], args['experimentname'])
    with track.trial(track_local_dir, track_remote_dir, param_map=args):
        track.debug("Starting trial")
        do_training(args)


if __name__ == '__main__':
    if args.config:
        print(">>> Using ray to launch experiments from config")
        raise NotImplementedError()
    else:
        experiment(args)
