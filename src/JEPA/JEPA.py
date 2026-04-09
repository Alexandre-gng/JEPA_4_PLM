import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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
    def __init__(self, input_dim, enc_hidden_dim: int = 64, latent_dim: int = 128, output_dim: int = 128, tau: float = 0.99):
        super(JEPA, self).__init__()
        self.tau = tau

        # enc(x) => Used to encode the entire protein sequence
        self.context_encoder = nn.Sequential(
            nn.Conv2d(input_dim, enc_hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        )

        # masked_enc(x) => used to encode the incomplete protein sequence (i.e. the masked sequence)
        self.target_encoder = nn.Sequential(
            nn.Conv2d(input_dim, enc_hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        )

        # Dec(z) => Used to decode the latent representation (vector) back to the real data
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim)
        )
    
    def update_target_encoder(self):
        """Update the target encoder's weights with the EMA
        """
        for param_context, param_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = self.tau * param_target.data + (1 - self.tau) * param_context.data

    def forward(self, full_sequence, sequence):

        # Encode the full sequence and keep local 8x8 latent representations
        # enc(full_sequence): (B, C, 8, 8) -> (B, 64, C)
        target_latent_map = self.context_encoder(full_sequence)
        target_latent = target_latent_map.flatten(2).transpose(1, 2)

        # Encode the masked sequence and keep local 8x8 latent representations
        # masked_enc(masked_sequence): (B, C, 8, 8) -> (B, 64, C)
        masked_latent_map = self.target_encoder(sequence)
        masked_latent = masked_latent_map.flatten(2).transpose(1, 2)

        # Predict each local latent representation from masked local latents
        # pred(masked_latent): (B, 64, C) -> (B, 64, output_dim)
        context_latent = self.predictor(masked_latent)
        
        return context_latent, target_latent