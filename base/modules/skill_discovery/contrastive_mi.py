import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule
import torch.nn.functional as F

class Discriminator(nn.Module, IntrinsicMotivationModule):
    def __init__(self, skill_number, state_size, hidden_size, num_layers=4, normalize_inputs=False,
                 input_key='next_state', input_size=None, temperature=0.5, traj_length_each_episode=50):
        super().__init__()
        self.temperature = temperature
        self.n = skill_number
        self.traj_length_each_episode = traj_length_each_episode
        self.state_size = int(state_size)  if input_size is None else int(input_size)
        self.input_key = str(input_key)
        assert num_layers >= 2
        self.num_layers = int(num_layers)
        self.count = 0 
        self.hidden_size = hidden_size
        self.normalize_inputs = normalize_inputs
        self.input_normalizer = Normalizer(self.state_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=self.state_size, output_size=self.n, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, input_normalizer=self.input_normalizer) # Q: why feature dim = skill_n?

        self.loss = self.compute_info_nce_loss
        
    def forward(self, batch):
        x = batch[self.input_key] # Q: next state or state?
        for layer in self.layers:
            x = layer(x)

        if 'positive' in batch.keys():
            y = batch['positive']
            for layer in self.layers:
                y = layer(y)

            return self.compute_cl_loss(x, y, labels=batch['skill']).mean() # , labels=batch['skill']

        return self.loss(x, batch['skill']).mean()
    
    def surprisal(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)

        if 'positive' in batch.keys():
            y = batch['positive']
            for layer in self.layers:
                y = layer(y)

            return torch.exp(-self.compute_cl_loss(x, y, labels=batch['skill'])).squeeze() # 

        return torch.exp(-self.compute_surprisal(x, batch['skill'])).squeeze() # CHANGE: seperate ir and loss computation
    


    def compute_cl_loss(self, features, positive_features, labels=None,):
        # BUG: TODO: Should dot product with itself not positive
        features = F.normalize(features, dim=1)
        positive_features = F.normalize(positive_features, dim=1)

        # Compute logits (scaled dot product)
        logits = torch.matmul(features, positive_features.T) / self.temperature  # shape: (B, B)

        # Labels: positive samples are diagonal
        classes = torch.arange(logits.size(0), device=logits.device)

        if labels is not None:
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).fill_diagonal_(0).bool().to(features.device)
            logits.masked_fill_(mask, float('-inf'))

        # Contrastive loss (InfoNCE)
        loss = F.cross_entropy(logits, classes, reduction="none").view(features.size(0), -1)

        return loss
    
    def compute_info_nce_loss(self, features, labels):
        # CHANGE: to device
        # label positives samples
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long().to(features.device) # (b, b) # NOTE: if the label the same matrix
        features = F.normalize(features, dim=1) # (b, c)
        similarity_matrix = torch.matmul(features, features.T) # (b, b)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device) # (b, b)
        labels = labels[~mask].view(labels.shape[0], -1) # (b, b - 1) # you use labels below, no need?
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (b, b - 1)

        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None] # NOTE: ([2500, 1])
        similarity_matrix = torch.exp(similarity_matrix)

        # another_positive = features.view(-1, self.traj_length_each_episode, self.n)[:, self.traj_length_each_episode-1, :].view(-1, self.n) # NOTE: (not ganranteed) choose the last step in each eps as the positve  # (goal_num, feature_dim) # self.n is num_skill
        # feature_dim = features.shape[-1]
        # another_positive = another_positive.flatten()   # (feature_dim * goal_num)
        # another_positive = another_positive.expand(self.traj_length_each_episode, -1)   # NOTE: should be self.traj_length_each_episode,     #(50, feature_dim * goal_num) # NOTE: 50 env.n
        # another_positive = another_positive.reshape(another_positive.shape[0], -1, feature_dim)        # (50, goal_num, feature_dim)
        # another_positive = another_positive.permute(1,0,2)      # (goal_num, 50, feature_dim) # NOTE: goal_num is the num of trajs
        # another_positive = another_positive.reshape(-1,feature_dim)    #(50 * goal_num (b), feature_dim) # NOTE: (goal1_feature*50||)
        

        # positives = torch.sum(torch.exp(features * another_positive), dim=-1, keepdim=True) # NOTE: should be dot product Q: goal*goal
        # negatives = torch.sum(similarity_matrix * (~labels.bool()).float(), dim=-1, keepdim=True) # Q: negative set megatitude changes? 

        # eps = torch.as_tensor(1e-6)
        # loss = -torch.log(positives / (negatives + eps) + eps) # Q: positive?

        # CHANGE: fix info nce computation
        random_max = labels*(1+torch.rand_like(labels, dtype=torch.float)) # .float()
        pick_one_positive_sample_idx = torch.argmax(random_max, dim=-1, keepdim=True)
        pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)
        
        positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True) #(b,1)
        # negatives = torch.sum(similarity_matrix, dim=-1, keepdim=True)  #(b,1)
        negatives = torch.sum(similarity_matrix * (~labels.bool()).float(), dim=-1, keepdim=True)  #(b,1)
        # CHANGE: Discard all positives and add back picked one
        eps = torch.as_tensor(1e-6)
        # loss = -torch.log(positives / (negatives + eps) + eps) #(b,1)
        loss = -torch.log(positives / (positives + negatives + eps) + eps) #(b,1)

        return loss
    
    def compute_surprisal(self, features, labels):
        # TODO: self.traj_length_each_episode -> self.skill_update_freq
        # CHANGE: to device
        # label positives samples

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long().to(features.device) # (b, b) # NOTE: if the label the same matrix
        features = F.normalize(features, dim=1) # (b, c)
        similarity_matrix = torch.matmul(features, features.T) # (b, b)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device) # (b, b)
        labels = labels[~mask].view(labels.shape[0], -1) # (b, b - 1) # you use labels below, no need?
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (b, b - 1)

        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None] # NOTE: ([2500, 1])
        similarity_matrix = torch.exp(similarity_matrix)

        another_positive = features.view(-1, self.traj_length_each_episode, self.n)[:, self.traj_length_each_episode-1, :].view(-1, self.n) # NOTE: (not ganranteed) choose the last step in each eps as the positive  # (goal_num, feature_dim) # self.n is num_skill
        feature_dim = features.shape[-1]
        another_positive = another_positive.flatten()   # (feature_dim * goal_num)
        another_positive = another_positive.expand(self.traj_length_each_episode, -1)   # CHANGE: should be self.traj_length_each_episode,     #(50, feature_dim * goal_num) # NOTE: 50 env.n
        another_positive = another_positive.reshape(another_positive.shape[0], -1, feature_dim)        # (50, goal_num, feature_dim)
        another_positive = another_positive.permute(1,0,2)      # (goal_num, 50, feature_dim) # NOTE: goal_num is the num of trajs
        another_positive = another_positive.reshape(-1,feature_dim)    #(50 * goal_num (b), feature_dim) # NOTE: (goal1_feature*50||)
        

        positives = torch.sum(torch.exp(features * another_positive), dim=-1, keepdim=True) # NOTE: should be dot product Q: goal*goal
        # positives = torch.exp(torch.sum(features * another_positive, dim=-1, keepdim=True))
        negatives = torch.sum(similarity_matrix * (~labels.bool()).float(), dim=-1, keepdim=True) # Q: negative set megatitude changes? 

        eps = torch.as_tensor(1e-6)
        # loss = -torch.log(positives / (negatives + eps) + eps) # Q: positive?
        loss = -torch.log(positives / (positives + negatives + eps) + eps) # Q: positive?

        # # CHANGE: fix info nce computation
        # pick_one_positive_sample_idx = torch.arange(labels.shape[0]).view(-1, 1).to(features.device)
        # pick_one_positive_sample_idx = self.traj_length_each_episode*(pick_one_positive_sample_idx//self.traj_length_each_episode + 1)-2
        # # pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        # pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)
        
        # positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True) #(b,1)
        # negatives = torch.sum(similarity_matrix, dim=-1, keepdim=True)  #(b,1)
        # eps = torch.as_tensor(1e-6)
        # loss = -torch.log(positives / (negatives + eps) + eps) #(b,1)



        return loss

