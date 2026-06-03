import torch
import torch.nn.functional as F
from torch import nn as nn
from torch import optim as optim
import math



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # To handle the output of the multi-head attention
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        # Q, K, V should have shape (batch, seq_len, d_model)
        batch_size = Q.size(0)
        q_len = Q.size(1)
        k_len = K.size(1)

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = Q.view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)

        att = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, V)

        y = y.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.W_o(y)

