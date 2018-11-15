""" Various utils for random seeding, progress, and metric tracking """
import collections
import hashlib
import numpy as np
import os
import shutil
import sys
import time

TOTAL_BAR_LENGTH = 65.

term_width = shutil.get_terminal_size().columns

def _next_seeds(n):
    # deterministically generate seeds for envs
    # not perfect due to correlation between generators,
    # but we can't use urandom here to have replicable experiments
    # https://stats.stackexchange.com/questions/233061
    mt_state_size = 624
    seeds = []
    for _ in range(n):
        state = np.random.randint(2**32, size=mt_state_size)
        digest = hashlib.sha224(state.tobytes()).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)[0]
        seeds.append(int(seed))
        if seeds[-1] is None:
            seeds[-1] = int(state.sum())
    return seeds


def seed_all(seed, numpy=True, random=True, torch=True):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    if numpy:
        np.random.seed(seed)
    rand_seed, torch_cpu_seed, torch_gpu_seed = _next_seeds(3)
    if random:
        import random
        random.seed(rand_seed)
    if torch:
        import torch
        torch.manual_seed(torch_cpu_seed)
        torch.cuda.manual_seed_all(torch_gpu_seed)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from
           https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RollingAverageWindow:
    """Creates an automatically windowed rolling average."""
    def __init__(self, window_size):
        self._window_size = window_size
        self._items = collections.deque([], window_size)
        self._total = 0

    def update(self, value):
        """updates the rolling window"""
        if len(self._items) < self._window_size:
            self._total += value
            self._items.append(value)
        else:
            self._total -= self._items.popleft()
            self._total += value
            self._items.append(value)

    def value(self):
        """returns the current windowed avg"""
        return self._total / len(self._items)
