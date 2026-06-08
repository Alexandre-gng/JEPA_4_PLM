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
        self.hidden_size = self.model.config.hidden_size

    def forward(self, sequence):
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def predict(self, sequence):
        return self.forward(sequence)
