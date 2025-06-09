# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from .base import StochasticAgent
from agents.maze_agents.toy_maze.env import Env
from base.modules.generic import OneHotEmbedding
from agents.maze_agents.modules import StochasticPolicy, Value
from base.modules.skill_discovery.endpoint_contrastive_mi import EndpointDiscriminator
from base.learners.skill_discovery.contrastive_mi import BaseContrastiveMILearner
from agents.maze_agents.toy_maze.skill_discovery.contrastive_mi import ContrastiveMILearner


class EndpointContrastiveMILearner(ContrastiveMILearner):
    AGENT_TYPE = "EndpointContrastiveMI"

    def __init__(self, *args, mode=None, use_reg=False, **kwargs): 
        self.use_reg = use_reg
        self.mode = mode
        if self.mode:  # default is ""
            assert self.mode in ["strict", "strict_s+", "strict_s+_s-", "strict_s-_s+"]
        super().__init__(*args, **kwargs)

    def _make_im_modules(self):
        return EndpointDiscriminator(self.skill_n, self._dummy_env.state_size,
                             mode=self.mode, use_reg=self.use_reg, num_layers=self.num_layers, hidden_size=self.hidden_size,
                             normalize_inputs=self.normalize_inputs, input_key='initial_terminal', input_size=self._dummy_env.state_size*2, **self.im_kwargs).to(self.device)

    def sample_positives(self, batched_episode):
        # TODO: device
        B = batched_episode["next_state"].size(0)
        batched_segments = {}
        initial_idx = [*range(0, B, self.skill_update_freq)]
        terminal_idx = [*range(self.skill_update_freq-1, B, self.skill_update_freq)]
        if len(terminal_idx) < len(initial_idx):
            terminal_idx.append(B-1)
        batched_segments['initial_terminal'] = torch.cat([batched_episode["state"][initial_idx],batched_episode["next_state"][terminal_idx]], dim=-1)
        batched_segments['skill'] =  batched_episode["skill"][initial_idx]
        labels = batched_segments["skill"]
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag = torch.eye(labels.shape[0], dtype=torch.bool)
        labels[diag] = 0
        labels = labels.long()

        random_max = (labels) * (1 + torch.rand_like(labels, dtype=torch.float))
        pick_one_positive_sample_idx = torch.argmax(random_max, dim=-1)
        batched_segments["positive"] = batched_segments["initial_terminal"][pick_one_positive_sample_idx]

        return batched_segments
    
    def relabel_episode(self, for_aux=False):
        # CHANGE: Comment out assert
        # assert len(self._compress_me[0]) == 50 * 50 and len(self._compress_me) == 1
        batched_episodes = []
        for ep in self._compress_me:
            batched_episode = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
            B = batched_episode["next_state"].size(0)
            # batched_episode = self.sample_positives(batched_episode)
            batched_episodes.append(batched_episode)

            if not for_aux:
                with torch.no_grad():
                    surprisals = self._compute_surprisal(batched_episode)

                surprisals = surprisals.view(-1, 1).expand(-1, self.skill_update_freq).flatten()[:B]

                if self.im_scale:
                    self.train()
                    _ = self._im_bn(surprisals.view(-1, 1))
                    self.eval()
                    surprisals = surprisals / torch.sqrt(self._im_bn.running_var[0])

                for e, s in zip(ep, surprisals):
                    e['reward'] += (self.im_nu * s.detach())
                    e['im_reward'] = s.detach()

        if for_aux:
            keys = batched_episodes[0].keys()
            batched_ep = {
                    k: torch.cat([b_ep[k] for b_ep in batched_episodes]) for k in keys
                }
            return batched_ep
        