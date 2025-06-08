# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
import torch.distributed as dist
from base.learners.base import BaseLearner


def ppo_decorator(partial_agent_class):
    assert issubclass(partial_agent_class, BaseLearner)

    class NewClass(partial_agent_class):

        def __init__(
            self,
            clip_range=0.2,
            horizon=None,
            mini_batch_size=None,
            rollouts=None,
            n_mini_batches=None,
            entropy_lambda=0.0,
            gae_lambda=0.98,
            aux_rollouts=None,
            **kwargs,
        ):

            if rollouts is None: # NOTE: rollouts=50
                assert horizon is not None
                assert mini_batch_size is not None
                self.horizon = int(horizon)
                self.mini_batch_size = int(mini_batch_size)
                self.rollouts = None
                self.n_mini_batches = None
            else:
                assert horizon is None
                assert mini_batch_size is None
                assert n_mini_batches is not None
                self.horizon = None
                self.mini_batch_size = None
                self.rollouts = int(rollouts)
                self.n_mini_batches = int(n_mini_batches)

            self.aux_rollouts = aux_rollouts  # CHANGE: Add aux rollout

            self.clip_range = clip_range
            self.entropy_lambda = float(entropy_lambda)
            self.gae_lambda = float(gae_lambda)

            self._mini_buffer = {}  # CHANGE: 'state': None
            self._epoch_transitions = {}
            self._batched_ep = None

            super().__init__(**kwargs)

            self.ep_summary_keys += ["avg_batch_rew", "avg_v", "v_loss", "p_loss", "e_loss"]

        @property
        def current_horizon(self):
            # CHANGE: state not necessary in buffer
            for k in self._mini_buffer.keys():
                mb_state = self._mini_buffer.get(k, None)
                if mb_state is None:
                    return 0
                else:
                    return mb_state.shape[0]

            return 0

        def add_to_mini_buffer(self, batched_episode):
            for k, v in batched_episode.items():
                if self._mini_buffer.get(k, None) is None:
                    self._mini_buffer[k] = v.detach()
                else:
                    self._mini_buffer[k] = torch.cat([self._mini_buffer[k], v], dim=0)

            curr_horizon = int(self.current_horizon)
            # CHANGE: avoid cleared key  if v is not None
            assert all([int(v.shape[0]) == curr_horizon for v in self._mini_buffer.values()])

        def fill_epoch_transitions(self):
            if self.horizon is not None:
                curr_horizon = int(self.current_horizon)
                assert curr_horizon >= self.horizon
                self._epoch_transitions = {}
                for k, v in self._mini_buffer.items():
                    self._epoch_transitions[k] = v[:self.horizon]
                    if curr_horizon > self.horizon:
                        self._mini_buffer[k] = v[self.horizon:]
                    else:
                        self._mini_buffer[k] = None
            else:
                curr_horizon = int(self.current_horizon)
                assert curr_horizon >= self.n_mini_batches # Q: n_mini_batches?
                self._epoch_transitions = {}
                for k, v in self._mini_buffer.items():
                    self._epoch_transitions[k] = v.detach() # if v is not None else None # CHANGE: possible none
                    # CHANGE: reset whole buffer
                    # self._mini_buffer[k] = None
                self._mini_buffer = {}

        def make_epoch_mini_batches(self, normalize_advantage=False):
            if self.horizon is not None:
                mb_indices = np.split(np.random.permutation(self.horizon), self.horizon // self.mini_batch_size)
            else:
                sz = [v.shape[0] for v in self._epoch_transitions.values()][0]
                # CHANGE: NOTE: Should not affect the orignal computation with positive
                # n_total = self.n_mini_batches * (sz // self.n_mini_batches)
                # perm_indices = np.random.permutation(sz)[:n_total]
                # mb_indices = np.split(perm_indices, self.n_mini_batches)

                traj_len = self.skill_update_freq
                seg_num = sz // traj_len # NOTE: ignore the incomplete
                n_total = self.n_mini_batches * (seg_num // self.n_mini_batches)
                perm_indices = np.random.permutation(seg_num)[:n_total]
                mb_indices = np.split(perm_indices, self.n_mini_batches) 
                # NOTE: n_mini_batches*desired_traj_num*skill_update_freq = self.rollouts * self.agent.env.n
                # 10*250(=5*50)*5=250*50
                # 10*50*5=50*50

            mini_batches = []
            for indices in mb_indices:
                # this_batch = {k: v[indices] for k, v in self._epoch_transitions.items()}
                this_batch = {k: torch.cat([v[i*traj_len:(i+1)*traj_len] for i in indices], dim=0) for k, v in self._epoch_transitions.items()}

                if normalize_advantage and 'advantage' in this_batch:
                    mb_mean = this_batch['advantage'].mean(dim=0, keepdim=True)
                    mb_std = this_batch['advantage'].std(dim=0, keepdim=True)
                    this_batch['advantage'] = (this_batch['advantage'] - mb_mean) / mb_std
                mini_batches.append(this_batch)
            return mini_batches

        def distributed_advantage_normalization(self):
            if 'advantage' not in self._epoch_transitions:
                return
            a = self._epoch_transitions['advantage']

            a_sum = a.sum(dim=0)
            a_sumsq = torch.pow(a, 2).sum(dim=0)
            dist.all_reduce(a_sum)
            dist.all_reduce(a_sumsq)
            # CHANGE: to device
            n = a.shape[0] * torch.ones(1, device=a.device)
            dist.all_reduce(n)
            # n = dist.get_world_size() * self.horizon

            a_mean = a_sum / n
            a_var = (a_sumsq / n) - (a_mean ** 2)
            a_std = torch.pow(a_var, 0.5) + 1e-8

            n_dim = len(a.shape)
            if n_dim == 1:
                self._epoch_transitions['advantage'] = (a - a_mean) / a_std
            elif n_dim == 2:
                self._epoch_transitions['advantage'] = (a - a_mean.view(1, -1)) / a_std.view(1, -1)
            else:
                raise NotImplementedError

        def reach_horizon(self, for_aux=False, *args, **kwargs):
            # Play until a certain number of transitions have been reached
            if self.horizon is not None:
                while self.current_horizon < self.horizon:
                    self.play_episode(*args, **kwargs)
                    batched_episode = {k: v.detach() for k, v in self.compress_episode().items()}
                    self.add_to_mini_buffer(batched_episode)

            # Play a specific number of rollouts
            else:
                rollout_num = self.rollouts if not for_aux else self.aux_rollouts
                try:
                    self.reset_ep_stats()
                except:
                    None
                for _ in range(rollout_num):
                    self.reset_ep_stats = self.play_episode(*args, **kwargs) # Q:return function pointer? # NOTE:add self.agent.episode to _compress_me

                if not for_aux:
                    self.relabel_episode(for_aux=for_aux)
                    batched_episode = {
                        k: v.detach() for k, v in self.compress_episode().items()
                    }  # NOTE: batch_ep
                else:
                    batched_episode = self.relabel_episode(for_aux=for_aux)

                self.add_to_mini_buffer(batched_episode)
            self.fill_epoch_transitions()  # NOTE: from mini_buffer to _epoch_transitions

        def _batch_episode(self, ep):
            keys = self.batch_keys
            if 'positive' in ep[0].keys():
                keys = self.batch_keys + ['positive']
            batched_episode = {key: torch.stack([e[key] for e in ep]) for key in keys} # NOTE: remove im_reward and env_reward

            batched_episode['value'] = self.get_values(batched_episode) # NOTE: compute value using v_module

            advs = torch.zeros_like(batched_episode['reward'])
            last_adv = 0
            for t in reversed(range(advs.shape[0])):
                if t == advs.shape[0] - 1:
                    # Bootstrap from early terminal means bootstrap from terminal value when episode didn't complete
                    if self.bootstrap_from_early_terminal:
                        has_next = 1.0 - batched_episode['complete'][t]  # Is this a genuine terminal action?
                    else:
                        has_next = 0.0

                    next_value = self.get_terminal_values(batched_episode)
                else:
                    has_next = 1.0  # By our setup, this cannot be a terminal action
                    next_value = batched_episode['value'][t + 1]

                delta = batched_episode['reward'][t] + self.gamma * next_value * has_next - batched_episode['value'][t]
                advs[t] = delta + self.gamma * self.gae_lambda * has_next * last_adv
                last_adv = advs[t]

            batched_episode['advantage'] = advs.detach()
            batched_episode['cumulative_return'] = advs.detach() + batched_episode['value'].detach()

            just_one_step = batched_episode['reward'].shape[0] == 1

            for k, v in batched_episode.items():
                if k not in self.no_squeeze_list:
                    new_v = torch.squeeze(v)
                    if just_one_step:
                        new_v = new_v[None]
                    batched_episode[k] = new_v

            return batched_episode

        def compress_episode(self):

            batched_episodes = [self._batch_episode(ep) for ep in self._compress_me]

            if len(batched_episodes) == 1:
                batched_ep = batched_episodes[0]

            else:
                keys = self.batch_keys + ['value', 'advantage', 'cumulative_return']
                # CHANGE: add positive if any
                if 'positive' in batched_episodes[0].keys():
                    keys.append('positive')
                batched_ep = {
                    k: torch.cat([b_ep[k] for b_ep in batched_episodes]) for k in keys
                } # NOTE: merge multiple eps

            self._batched_ep = batched_ep 
            # NOTE: self._batched_ep = [('state', torch.Size([2500, 2])), ('next_state', torch.Size([2500, 2])), ('skill', torch.Size([2500])), ('action', torch.Size([2500, 2])), ('n_ent', torch.Size([2500])), ('log_prob', torch.Size([2500])), ('action_logit', torch.Size([2500, 2])), ('reward', torch.Size([2500])), ('terminal', torch.Size([2500])), ('complete', torch.Size([2500])), ('value', torch.Size([2500])), ('advantage', torch.Size([2500])), ('cumulative_return', torch.Size([2500]))]
            # NOTE: self._compress_me[0] = [dict(eps)]

            return batched_ep

        def episode_summary(self):
            if not self._batched_ep:
                _ = self.compress_episode()
            if not self._ep_summary:
                _ = self() # Q: repeat calling value function and policy?
            return [float(x) for x in self._ep_summary]

        def forward(self, mini_batch=None, return_stats=False):
            # CHANGE: return stats for logging
            if mini_batch is None:
                # We're here to get the stats
                mini_batch = self._batched_ep
                fill_summary = True
            else:
                # We're here to compute the
                fill_summary = False
                self.train()

            value = self.get_values(mini_batch) # NOTE:2500 
            assert value.shape == mini_batch['cumulative_return'].shape
            v_losses = 0.5 * torch.pow(mini_batch['cumulative_return'] - value, 2)
            v_loss = v_losses.mean()

            log_prob, n_ent = self.get_policy_lprobs_and_nents(mini_batch) # NOTE: a little bit repeat over act()
            e_loss = n_ent.mean()

            # Defining Loss = - J is equivalent to max J
            assert log_prob.shape == mini_batch['log_prob'].shape
            ratio = torch.exp(log_prob - mini_batch['log_prob'])

            pg_losses1 = -mini_batch['advantage'] * ratio
            pg_losses2 = -mini_batch['advantage'] * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            p_losses = torch.max(pg_losses1, pg_losses2)
            p_loss = p_losses.mean()

            loss = v_loss + p_loss + (self.entropy_lambda * e_loss)

            if fill_summary:
                self.fill_summary(mini_batch['reward'].mean(), value.mean(), v_loss, p_loss, e_loss) # NOTE: add to _ep_summary a list of stats 

            self.eval() # Q: set eval here? before optim?

            if return_stats:
                return loss, {'agent_loss': loss.item(), 
                        'intr_reward': mini_batch['reward'].mean().item(),
                        'value_func': value.mean().item(),
                        'v_loss': v_loss.item(),
                        'p_loss': p_loss.item(),
                        'e_loss': e_loss.item(),
                        'log_prob': log_prob.mean().item()}

            return loss

    return NewClass
