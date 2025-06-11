import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule
from base.modules.skill_discovery.contrastive_mi import Discriminator
import torch.nn.functional as F


class EndpointDiscriminator(Discriminator):
    def __init__(self, *args, mode=None, use_reg=False, **kwargs):
        self.mode = mode
        if self.mode:  # default is ""
            assert self.mode in ["strict", "strict_s+", "strict_s+_s-", "strict_s-_s+", "strict_s-"]
        super().__init__(*args, **kwargs)

        if use_reg:
            self.reg_layers = create_nn(
                input_size=self.state_size,
                output_size=self.n,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                input_normalizer=self.input_normalizer,
            )
            self.reg_layers2 = create_nn(
                input_size=self.state_size // 2,
                output_size=self.n,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                input_normalizer=self.input_normalizer,
            )

    def forward(self, batch):
        loss, reg_loss = self.compute_cl_loss(batch)
        if reg_loss is not None:
            return loss.mean(), reg_loss.mean()
        else:
            return loss.mean()

    def surprisal(self, batch):
        loss, reg_loss = self.compute_cl_loss(batch)
        if reg_loss is not None:
            # TODO: change lambda later, because use whole eps, length should be the same
            return torch.exp(-loss) + 0.1*torch.exp(-reg_loss)
        else:
            return torch.exp(-loss)

    def compute_cl_loss(self, batch):
        # s0, st
        B = batch["next_state"].size(0)
        skill_len = self.traj_length_each_episode  # overload
        next_states = batch["next_state"]
        initial_idx = [*range(0, B, skill_len)]
        
        # if hasattr(self, 'reg_layers'):
        #     next_state_features = batch["next_state"]
        #     for layer in self.reg_layers:
        #         next_state_features = layer(next_state_features)
        #     # TODO: compute_info_nce_loss # strict true negtives only NOT strict for positives
        #     # compute_surprisal # s_t,g
        #     reg_loss = self.compute_info_nce_loss(next_state_features, batch["skill"]).squeeze() 
        # else:
        #     reg_loss = None

        initials = batch["state"][initial_idx]
        expanded_initials = (
            torch.unsqueeze(initials, dim=1).expand(-1, skill_len, -1).reshape(-1, initials.shape[-1])
        )
        features = torch.cat([expanded_initials, next_states], dim=-1)

        if hasattr(self, 'reg_layers'):
            terminal_idx = [*range(skill_len - 1, B, skill_len)]  # assume dividable
            if len(terminal_idx) < len(initial_idx):
                terminal_idx.append(B - 1)
            terminals = batch["next_state"][terminal_idx]
            expanded_terminals = (
                torch.unsqueeze(terminals, dim=1).expand(-1, skill_len, -1).reshape(-1, terminals.shape[-1])
            )
            reg_loss = self.compute_reg_loss(features, labels=batch["skill"], positives=expanded_terminals)
        else:
            reg_loss = None

        for layer in self.layers:
            features = layer(features)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        # similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None]

        assert not(torch.isnan(similarity_matrix).any() or torch.isinf(similarity_matrix).any())

        labels = batch["skill"]
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = torch.eye(labels.shape[0], dtype=torch.bool)

        t_mask = torch.arange(labels.shape[0]).to(labels.device)
        t_mask = t_mask % skill_len
        t_mask = t_mask.unsqueeze(0) == t_mask.unsqueeze(1)


        similarity_matrix.masked_fill_(self_mask, float("-inf"))
        candidate_mask = (labels & t_mask & (~self_mask)).int()
        # NOTE: len is not B now
        has_postive = torch.sum(candidate_mask, dim=-1)!=0
        candidate_mask = candidate_mask[has_postive]
        similarity_matrix = similarity_matrix[has_postive]
        random_max = candidate_mask * (1 + torch.rand_like(candidate_mask, dtype=torch.float, device=candidate_mask.device))
        pick_one_positive_sample_idx = torch.argmax(random_max, dim=-1, keepdim=True)

        classes = pick_one_positive_sample_idx.squeeze()
        # classes = torch.arange(labels.size(1), device=labels.device)[initial_idx]

        loss = F.cross_entropy(similarity_matrix, classes, reduction="none")
        

        return loss, reg_loss

    def compute_reg_loss(self, features, labels, positives=None, ):
        # NOTE: features are states
        for layer in self.reg_layers:
            features = layer(features)

        if positives is None:
            positives = features
        else:
            for layer in self.reg_layers2:
                positives = layer(positives)

        features = F.normalize(features, dim=1)
        positives = F.normalize(positives, dim=1)

        # Compute logits (scaled dot product)
        logits = torch.matmul(features, positives.T) / self.temperature  # shape: (B, B)
        assert not(torch.isnan(logits).any() or torch.isinf(logits).any())

        # Labels: positive samples are diagonal
        classes = torch.arange(logits.size(0), device=logits.device)

        # mask out negative pairs
        if self.mode == 'strict':
            mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).to(features.device)
            logits.masked_fill_(mask, float('-inf'))

        # Contrastive loss (InfoNCE)
        loss = F.cross_entropy(logits, classes, reduction="none")

        return loss





    # def compute_cl_loss(self, batch):
    #     # s0, g
    #     B = batch["next_state"].size(0)
    #     skill_len = self.traj_length_each_episode  # overload
    #     states = batch["state"]
    #     initial_idx = [*range(0, B, skill_len)]
    #     terminal_idx = [*range(skill_len - 1, B, skill_len)]  # assume dividable
    #     if len(terminal_idx) < len(initial_idx):
    #         terminal_idx.append(B - 1)
        
    #     if hasattr(self, 'reg_layers'):
    #         terminals = batch["next_state"][terminal_idx]
    #         terminal_labels = batch["skill"][terminal_idx]
    #         for layer in self.reg_layers:
    #             terminals = layer(terminals)
    #         reg_loss = self.compute_info_nce_loss(terminals, terminal_labels).squeeze() # strict true negtives only
    #     else:
    #         reg_loss = None

    #     terminals = batch["next_state"][terminal_idx]
    #     expanded_terminals = (
    #         torch.unsqueeze(terminals, dim=1).expand(-1, skill_len, -1).reshape(-1, terminals.shape[-1])
    #     )
    #     state_terminals = torch.cat([states, expanded_terminals], dim=-1)

    #     for layer in self.layers:
    #         state_terminals = layer(state_terminals)

    #     state_terminals = F.normalize(state_terminals, dim=1)
    #     initial_terminals = state_terminals[initial_idx]
    #     similarity_matrix = torch.matmul(initial_terminals, state_terminals.T) / self.temperature
        
    #     assert not(torch.isnan(similarity_matrix).any() or torch.isinf(similarity_matrix).any())
            

    #     labels = batch["skill"]
    #     labels = labels.unsqueeze(0) == labels.unsqueeze(1)
    #     diag = torch.eye(labels.shape[0], dtype=torch.bool)
    #     labels[diag] = 0
    #     # other s
    #     s_mask = torch.arange(labels.shape[0], dtype=int).to(labels.device)
    #     s_mask = s_mask // skill_len
    #     s_mask = s_mask.unsqueeze(0) == s_mask.unsqueeze(1)
    #     s_mask = s_mask[initial_idx]
    #     labels = labels[initial_idx]
    #     self_mask = diag[initial_idx]
    #     initial_mask = torch.zeros_like(labels, dtype=bool).to(labels.device)
    #     initial_mask[:, torch.arange(0, B, skill_len, dtype=int)] = True

    #     if self.mode == "strict":
    #         # strict: negative initial_terminal pairs only
    #         mask = torch.logical_or(~initial_mask, labels)
    #     elif self.mode == "strict_s+_s-":
    #         # mask out positive pairs from other trajectories
    #         mask = (~s_mask) & labels
    #     elif self.mode == "strict_s-_s+":
    #         # mask out negative non initial_terminal pairs
    #         mask = torch.logical_or((~s_mask) & (~labels) & (~initial_mask), labels & initial_mask)
    #         # # TODO: NOTE: TEMP: Check pos/neg ration effect
    #         # true_negative_mask = initial_mask & (~labels)
    #         # true_negative_mask[:, : mask.shape[1] // 2] = False
    #         # mask = torch.logical_or(mask, true_negative_mask)
    #     elif self.mode == "strict_s+":
    #         # besides strict, add positive pairs from the same trajectories
    #         mask = (~s_mask) & torch.logical_or(labels, ~initial_mask)
    #     else:
    #         # default: mask out positive initial_terminal pairs
    #         mask = labels & initial_mask

    #     candidate_mask = (labels & initial_mask & (~self_mask)).int()
    #     has_no_postive = torch.sum(candidate_mask, dim=-1)==0
    #     # TODO: is a skill has no positve
    #     assert torch.sum(has_no_postive.int()) <= 5*skill_len
    #     if has_no_postive.any():
    #         candidate_mask[has_no_postive, -1] = 1
    #     random_max = candidate_mask * (1 + torch.rand_like(labels, dtype=torch.float, device=labels.device))
    #     pick_one_positive_sample_idx = torch.argmax(random_max, dim=-1, keepdim=True)
    #     candidate_mask = torch.zeros_like(labels, device=labels.device).scatter_(
    #         -1, pick_one_positive_sample_idx, 1
    #     )
    #     mask = torch.logical_or(mask, self_mask) & (~candidate_mask)

    #     similarity_matrix.masked_fill_(mask, float("-inf"))

    #     classes = pick_one_positive_sample_idx.squeeze()
    #     # classes = torch.arange(labels.size(1), device=labels.device)[initial_idx]

    #     loss = F.cross_entropy(similarity_matrix, classes, reduction="none")

    #     return loss, reg_loss
    