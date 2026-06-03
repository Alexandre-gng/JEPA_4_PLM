import torch
from torch import nn as nn

from transformers import AutoTokenizer, AutoModel

class ESM2_8M(nn.Module):
    """
    ESM2_8M as an Encoder for the JEPA model.
    """
    def __init__(self):
        super(ESM2_8M, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    def forward(self, sequence):
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def predict(self, sequence):
        return self.forward(sequence)