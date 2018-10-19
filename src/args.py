""" Handle all of the arguments parsing  """

import argparse
import os
from dotenv import load_dotenv, find_dotenv

parsed = {}

# Append the environment information if we want it
# It's all stored in the path now
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
parsed['projectname'] = os.environ.get('projectname')
parsed['dataroot'] = os.environ.get('dataroot')
parsed['remote'] = os.environ.get('remote')

# Create the parser
parser = argparse.ArgumentParser(description='parser for %s'
                                 % parsed['projectname'])

# Generic arguments 
parser.add_argument('experimentname', help='Name of the experiment to run')

# Ray arguments
parser.add_argument('self_host', default=0, help='if > 0, create ray host '
                    'with specified number of GPUs')
parser.add_argument('cpu', action='store_true', help='use cpu only')
parser.add_argument('port', default=6379, type=int, help='ray port')
parser.add_argument('server_port', default=10000, type=int,
                    help='ray tune port')
parser.add_argument('--config', default='')

# Continue with parsing!
_is_parsed = False


def get_parser():
    global _is_parsed
    assert not _is_parsed, "Can't get parser after parsing!"
    return parser


def parse():
    global _is_parsed
    if not _is_parsed:
        _parsed = parser.parse_args()
        global parsed
        parsed.extend(vars(_parsed))
        _is_parsed = True
    return parsed

