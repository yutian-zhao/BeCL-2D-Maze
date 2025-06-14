# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch.distributed as dist
from dist_train.workers import baseline
import numpy as np 
import torch
import random

episodic_off_policy_manager_lookup = {
    'baseline': baseline.EpisodicOffPolicy,
    'hierarchical': baseline.HierarchicalEpisodicOffPolicy
}

off_policy_manager_lookup = {
    'baseline': baseline.OffPolicy,
}

on_policy_manager_lookup = {
    'baseline': baseline.OnPolicy,
}

ppo_manager_lookup = {
    'baseline': baseline.PPO,
    'hierarchical': baseline.HierarchicalPPO
}

# For listing the current algorithms (see agents/base/algorithm_deecorators/) that belong to each manager group
on_policy_algos = []  # (ignore PPO here; it is unique)
off_policy_algos = ['sac']
episodic_off_policy_algos = ['ddpg', 'dqn']

def try_until_success(rank, settings, max_retries=10):
    attempts = 0
    while attempts < max_retries:
        try:
            ip = np.random.randint(10,99)
            dist.init_process_group(
                backend='gloo',
                init_method='tcp://127.0.0.1:432{}'.format(str(ip)),
                rank=rank,
                world_size=settings.N
            ) 
            return "432" + str(ip)
        except Exception as e:
            attempts += 1
            if attempts >= max_retries:
                raise e

def synchronous_worker(rank, config, settings):
    """Create a worker to play episodes on a given port and send the results to the trainer"""
    # Create a distributed process so the workers can share gradients and other such things
    # CHANGE:
    ip = try_until_success(rank, settings)

    # Create a distributed process so the workers can share gradients and other such things
    # dist.init_process_group(
    #     backend='gloo',
    #     init_method=f'tcp://127.0.0.1:{settings.port}',
    #     rank=rank,
    #     world_size=settings.N
    # )

    if 'seed' in config.keys():
        seed = config['seed']
        print(f"Set random seed to {seed}")
        torch.manual_seed(seed)
        if config.get('device', None) == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    print(f'Rank {rank} worker successfully initiated the distributed process group at port {ip}!', flush=True)

    train_type = config['train_type']

    style = 'hierarchical' if 'hierarchical' in config['learner_type'].lower() else 'baseline'

    # Training is managed according to the PPO set up
    if train_type == 'ppo':
        manager_class = ppo_manager_lookup[style]

    # Doing some on-policy learning algorithm
    elif train_type in on_policy_algos:
        manager_class = on_policy_manager_lookup[style]

    # Doing some off-policy learning algorithm
    elif train_type in off_policy_algos:
        manager_class = off_policy_manager_lookup[style]

    # Doing some (episodic) off-policy learning algorithm
    elif train_type in episodic_off_policy_algos:
        manager_class = episodic_off_policy_manager_lookup[style]

    else:
        raise ValueError('Could not associate train_type "{}" with any known training manager'.format(train_type))

    # Create a manager object for this worker
    manager = manager_class(rank, config, settings)

    # Run through however many epochs we're supposed to
    for _ in range(int(settings.dur)):
         manager.do_epoch()
