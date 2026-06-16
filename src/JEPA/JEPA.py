import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from predictor.MultiHeadAttn import MultiHeadAttention

from encoder.ESM2_8M import ESM2_8M

@dataclass
class JEPAConfig:
    input_dim: int = 12
    enc_hidden_dim: int = 64
    latent_dim: int = 128
    output_dim: int = 128
    batch_size: int = 64
    patience: int = 5
    num_epochs: int = 10
    learning_rate: float = 1e-3
    tau: float = 0.99


class JEPA(nn.Module):
    """
    JEPA (Joint Embedding Predictive Architecture) for protein sequences.
    - enc(x) => Used to encode the entire protein sequence
    - masked_enc(x) => used to encode the incomplete protein sequence (i.e. the masked sequence)
    - Dec(z) => Used to decode the latent representation (vector) back to the real data
    """
    def __init__(self, latent_dim: int | None = None, output_dim: int = 320, tau: float = 0.99):
        super(JEPA, self).__init__()
        self.tau = tau

        self.context_encoder = ESM2_8M()
        self.target_encoder = ESM2_8M()

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

        self.encoder_dim = self.context_encoder.hidden_size
        self.latent_dim = latent_dim if latent_dim is not None else self.encoder_dim
        self.projection = (
            nn.Linear(self.encoder_dim, self.latent_dim)
            if self.latent_dim != self.encoder_dim
            else nn.Identity()
        )

        self.predictor = MultiHeadAttention(num_heads=1, d_model=self.latent_dim)


    @torch.no_grad()
    def update_target_encoder(self):
        for param_context, param_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.mul_(self.tau).add_(param_context.data, alpha=1 - self.tau)


    def forward(self, full_sequence, sequence):
        target_latent = self.context_encoder(full_sequence)
        target_latent = self.projection(target_latent)

        masked_latent = self.target_encoder(sequence)
        masked_latent = self.projection(masked_latent)

        context_latent = self.predictor(masked_latent, masked_latent, masked_latent)
        return context_latent, target_latent        
    
