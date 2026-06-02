import torch
import torch.nn.functional as F
from torch import nn as nn
from torch import optim as optim
import math


MAX_INPUT = 1000


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """
    def __init__(self, num_heads, d_model = MAX_INPUT):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        Q = self.W_q(Q).view(-1, self.num_heads, self.d_k)
        K = self.W_k(K).view(-1, self.num_heads, self.d_k)
        V = self.W_v(V).view(-1, self.num_heads, self.d_k)
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = F.softmax(att, dim=-1)
        y = att @ V
        return y.view(-1, self.d_model)