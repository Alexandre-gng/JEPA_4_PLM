import torch
from torch import nn as nn
from .MultiHeadAttn import MultiHeadAttention

class CrossAttention(nn.Module):
    """
    Cross-attention module
    """
    def __init__(self, d, n_heads=4):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(n_heads, d)
        
    def forward(self, x, y):
        return self.mha(x, y, y)