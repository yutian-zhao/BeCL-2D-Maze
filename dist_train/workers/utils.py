# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import time
import torch
import pickle
import logging
import numpy as np

# CHANGE: Logger class is from https://github.com/rll-research/url_benchmark/blob/main/logger.py
import csv
import datetime
from collections import defaultdict

# import numpy as np
# import torch
# import torchvision
import wandb
from termcolor import colored
# from torch.utils.tensorboard import SummaryWriter

COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('fps', 'FPS', 'float'), ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time')]

COMMON_FORMAT = [ ('agent_loss', 'AL', 'float'), 
                ('intr_reward', 'IR', 'float'),
                ('value_func', 'V', 'float'),
                ('v_loss', 'VL', 'float'),
                ('p_loss', 'PL', 'float'),
                ('e_loss', 'EN', 'float'),
                ('log_prob', 'LP', 'float'),
                ('contrastive_loss', 'CL', 'float')]


def create_worker_logger(worker_id, exp_dir, group_id=None):
    # create logger
    logger = logging.getLogger('{:02d}'.format(worker_id))
    logger.setLevel(logging.DEBUG)

    # # create file formatter
    # log_name = '{}.log'.format(worker_id) if group_id is None else '{}.{}.log'.format(worker_id, group_id)
    # fh = logging.FileHandler(filename=os.path.join(exp_dir, log_name))
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(name)s - %(message)s'))
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating, use_wandb):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None
        self.use_wandb = use_wandb

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def _dump_to_wandb(self, data):
        wandb.log(data)
        # self.run.log(data)

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        if self.use_wandb:
            wandb_data = {prefix + '/' + key: val for key, val in data.items()}
            self._dump_to_wandb(data=wandb_data)
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, rank, log_dir, use_tb, use_wandb):
        self._log_dir = log_dir
        self.rank = rank
        self._train_mg = MetersGroup(log_dir / f'train_{rank}.csv',
                                     formating=COMMON_FORMAT,
                                     use_wandb=use_wandb)
        self._eval_mg = MetersGroup(log_dir / f'eval_{rank}.csv',
                                    formating=COMMON_FORMAT,
                                    use_wandb=use_wandb)
        if use_tb:
            from torch.utils.tensorboard import SummaryWriter
            self._sw = SummaryWriter(str(log_dir / f'tb_{rank}'))
        else:
            self._sw = None
        self.use_wandb = use_wandb
        if self.use_wandb:
            dname = log_dir.parts[-1]
            self.run = wandb.init(project="2d-maze", group=str(dname[:-9]), name=str(dname[-8:])+f'_{rank}')

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)


class ReplayBuffer:
    def __init__(self, model, config, verbose_load=True):
        self.model = model

        if config.get('load_buffer', False):
            self._load_buffer(config, verbose=verbose_load)
        else:
            self.capacity = config['buffer_capacity']
            self.min_size = max(config['min_buffer_size'], config['batch_size'])
            self.ep_buffer = [{}] * self.capacity
            self._pointer = 0
            self._looped = False

        self.batch_size = config['batch_size']
        self.opt_count = [0] * self.capacity

        self.profiler = {
            'st': time.time(),
            'last_idle': time.time(),
            'time_idle': 0,
            'time_sync': 0,
            'time_batch': 0,
            'size': [],
            'opts': [],
        }

    def save_buffer(self, path):
        assert os.path.isdir(path), "save_buffer needs to be given an existing directory where files will be stored"

        # Save buffer-related variables
        state_dict = dict(capacity=self.capacity, min_size=self.min_size, _pointer=self._pointer, _looped=self._looped)
        with open(os.path.join(path, "state_dict.pkl"), 'wb') as f:
            pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)

        # Save the contents of the buffer, with one file per key
        keys = self.ep_buffer[0].keys()
        ep_idxs = np.arange(self.size)
        for k in keys:
            values = torch.stack([self.ep_buffer[i][k] for i in ep_idxs]).detach()
            filename = "{}.pt".format(k)
            torch.save(values, os.path.join(path, filename))

    def _load_buffer(self, config, verbose=True):
        assert 'buffer_path' in config, "'buffer_path needs to be set when load_buffer=True"
        state_dict_path = os.path.join(config['buffer_path'], 'state_dict.pkl')
        with open(state_dict_path, 'rb') as f:
            state_dict = pickle.load(f)
        self.capacity = state_dict['capacity']
        self.min_size = max(state_dict['min_size'], config['batch_size'])
        self._pointer = state_dict['_pointer']
        self._looped = state_dict['_looped']

        self.ep_buffer = [{} for _ in range(self.capacity)]
        pt_filenames = [f for f in os.listdir(config['buffer_path']) if f.endswith('.pt')]
        for filename in pt_filenames:
            k = filename[:-3]  # remove the ".pt" extension
            values = torch.load(os.path.join(config['buffer_path'], filename))
            for idx in range(self.size):
                self.ep_buffer[idx][k] = values[idx]

        if verbose:
            print("\n\nLoaded ReplayBuffer")
            print("  Path: {}".format(config['buffer_path']))
            print("  Size: {}".format(self.size))
            print("\n")

    def reset_profiler(self):
        self.profiler = {
            'st': time.time(),
            'last_idle': time.time(),
            'time_idle': 0,
            'time_sync': 0,
            'time_batch': 0,
            'size': [],
            'opts': [],
        }

    @property
    def size(self):
        return int(self.capacity) if self._looped else int(self._pointer)

    def add_episode(self, transition_dicts):
        """Integrate new episodes from the workers into the buffers"""
        self.profiler['time_idle'] += time.time() - self.profiler['last_idle']
        st = time.time()
        self.ep_buffer[self._pointer] = {}
        self.opt_count[self._pointer] = 0
        for t in transition_dicts:
            self.ep_buffer[self._pointer] = {k: v.detach() for k, v in t.items()}
            self._pointer += 1
            if self._pointer >= self.capacity:
                self._looped = True
                self._pointer = 0

        self.profiler['size'].append(self.size)
        self.profiler['time_sync'] += time.time() - st
        self.profiler['last_idle'] = time.time()

    def make_batch(self, normalize=True):
        """Convert the buffer into inputs for DataParallel training"""
        self.profiler['time_idle'] += time.time() - self.profiler['last_idle']
        st = time.time()
        keys = self.ep_buffer[0].keys()
        batch = dict()
        ep_idxs = np.random.permutation(np.arange(self.size))[:self.batch_size]
        for k in keys:
            batch[k] = torch.stack([self.ep_buffer[i][k] for i in ep_idxs]).detach()
        for i in ep_idxs:
            self.opt_count[i] += 1

        if normalize:
            # Apply normalization to the batch
            batch = self.model.normalize_batch(batch)

        # Profile time statistics
        self.profiler['time_batch'] += time.time() - st
        self.profiler['last_idle'] = time.time()

        return batch

    def profile(self, time_window=900.):
        time_window = float(time_window)
        dur = time.time() - self.profiler['st']
        if dur < time_window:
            return

        print('\nTrainer time expenditure:\n  Syncing: {:5.2f}%,  Batching: {:5.2f}%  Training: {:5.2f}%'.format(
            100*self.profiler['time_sync']/dur,
            100*self.profiler['time_batch']/dur,
            100*self.profiler['time_idle']/dur,
        ), flush=True)
        # print('Average buffer size: {:7.1f}'.format(np.mean(self.profiler['size'])), flush=True)
        # print('Average episode use: {:7.1f}'.format(np.mean(self.profiler['opts'])), flush=True)
        # print(' ', flush=True)
        print('Current buffer size: {:7.1f}\n'.format(self.size), flush=True)

        self.reset_profiler()