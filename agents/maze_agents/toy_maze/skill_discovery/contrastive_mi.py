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
    def create_env(self):
        return Env(**self.env_params)

    def _make_agent(self):
        return StochasticAgent(skill_n=self.skill_n, env=self.create_env(), policy=self.policy,
                               skill_embedding=self.skill_emb, device=self.device).to(self.device)

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
