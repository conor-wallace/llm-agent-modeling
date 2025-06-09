import copy
import numpy as np
import itertools
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, OneHotCategorical

from utils import onehot_from_logits


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MLPLayer, self).__init__()
        self._layer_N = 1

        active_func = nn.Tanh()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain('tanh')

        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_size), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), active_func, nn.LayerNorm(hidden_size))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, cat_self=True, attn_internal=False, embedding_size=None):
        super(MLPBase, self).__init__()

        self._layer_N = 1
        self.hidden_size = hidden_dim

        self.feature_norm = nn.LayerNorm(input_dim)

        self.mlp = MLPLayer(input_dim, self.hidden_size)

    def forward(self, x):
        x = self.feature_norm(x)
        x = self.mlp(x)

        return x


class PPONet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(PPONet, self).__init__()

        self.base = MLPBase(input_dim, hidden)
        self.act = nn.Linear(hidden, output_dim)
    
    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        x = self.base(x)
        logits = self.act(x)
        probs = Categorical(logits=logits).probs
        action = onehot_from_logits(probs)
        return action[0].detach().numpy()