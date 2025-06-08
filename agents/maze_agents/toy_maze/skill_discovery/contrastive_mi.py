# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from .base import StochasticAgent
from agents.maze_agents.toy_maze.env import Env
from base.modules.generic import OneHotEmbedding
from agents.maze_agents.modules import StochasticPolicy, Value
from base.modules.skill_discovery.contrastive_mi import Discriminator
from base.learners.skill_discovery.contrastive_mi import BaseContrastiveMILearner


class ContrastiveMILearner(BaseContrastiveMILearner):
    # CHANGE: to device
    # NOTE: create env and make agent/actor
    def __init__(self, *args, mode=None, **kwargs): 
        if not hasattr(self, 'mode'):
            self.mode = mode
            if self.mode: # default is ""
                assert self.mode in ["gs-", "gs+", "g", "s+", "s-", "s"]

        super().__init__(*args, **kwargs)

    def create_env(self):
        return Env(**self.env_params)

    def _make_agent(self):
        return StochasticAgent(skill_n=self.skill_n, env=self.create_env(), policy=self.policy,
                               skill_embedding=self.skill_emb, device=self.device, skill_update_freq=self.skill_update_freq).to(self.device)

    def _make_agent_modules(self):
        self._make_skill_embedding()
        kwargs = dict(env=self._dummy_env, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      goal_size=self.skill_n, normalize_inputs=self.normalize_inputs)
        self.policy = StochasticPolicy(**kwargs).to(self.device)
        self.v_module = Value(use_antigoal=False, **kwargs).to(self.device)

    def _make_skill_embedding(self):
        self.skill_emb = OneHotEmbedding(self.skill_n).to(self.device) # NOTE: Did not use module

    def _make_im_modules(self):
        return Discriminator(self.skill_n, self._dummy_env.state_size,
                             num_layers=self.num_layers, hidden_size=self.hidden_size,
                             normalize_inputs=self.normalize_inputs, **self.im_kwargs).to(self.device)

    def sample_positives(self, batched_episode):
        # sample positive from batched episodes
        # mode:
        # mode should be in ['gs-', 'gs+', 'g', 's+', 's-', 's']
        # s:(default) filter out states with  different labels
        # g: filter out non-terminal state
        # s+: filter out from the different trajectory/episode
        # s-: filter out states from the same trajectory/episode
        if self.mode == "gs+":  # need special treatmeet because otherwise g has no corresponding positive
            B = batched_episode["next_state"].size(0)
            idx = (torch.arange(B) // self.skill_update_freq + 1) * self.skill_update_freq - 1
            # TODO: should ignore incomplete skill
            pick_one_positive_sample_idx = torch.minimum(idx, torch.ones_like(idx) * (B - 1))
            batched_episode["positive"] = batched_episode["next_state"][pick_one_positive_sample_idx]
            return batched_episode

        else:
            labels = batched_episode["skill"]
            labels = labels.unsqueeze(0) == labels.unsqueeze(1)

            diag = torch.eye(labels.shape[0], dtype=torch.bool)  # (b, b)
            labels[diag] = 0

            g_mask = torch.zeros_like(labels, dtype=bool)
            g_mask[:, torch.arange(self.agent.env.n - 1, labels.shape[1], self.agent.env.n, dtype=int)] = True
            # other s
            s_mask = torch.arange(labels.shape[0], dtype=int)
            s_mask = s_mask // self.agent.env.n
            other_s_mask = s_mask.unsqueeze(0) != s_mask.unsqueeze(1)
            s_mask = s_mask.unsqueeze(0) == s_mask.unsqueeze(1)

            if "g" in self.mode:
                labels *= g_mask
            if "s+" in self.mode:
                labels *= s_mask
            if "s-" in self.mode:
                labels *= other_s_mask

            # default is random across all states with the same label
            labels = labels.long()
            # row_has_nonzero = candidate_mask.any(dim=1)
            # assert row_has_nonzero.all(), "Some rows have no non-zero elements"
            random_max = (labels) * (1 + torch.rand_like(labels, dtype=torch.float))
            pick_one_positive_sample_idx = torch.argmax(random_max, dim=-1)
            batched_episode["positive"] = batched_episode["next_state"][pick_one_positive_sample_idx]

            return batched_episode

    def relabel_episode(self, for_aux=False):
        # CHANGE: Comment out assert
        # assert len(self._compress_me[0]) == 50 * 50 and len(self._compress_me) == 1
        batched_episodes = []
        for ep in self._compress_me:
            batched_episode = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
            batched_episode = self.sample_positives(batched_episode)

            assert len(ep) == len(batched_episode["positive"])

            batched_episodes.append(batched_episode)

            # TODO: modify add_im_reward, no need
            for e, pos in zip(ep, batched_episode["positive"]):
                e["positive"] = pos

        if for_aux:
            keys = ["state", "next_state", "skill", "positive"]
            batched_ep = {k: torch.cat([b_ep[k] for b_ep in batched_episodes]) for k in keys}
            return batched_ep

        # Add discriminator reward
        else:
            self._add_im_reward()
