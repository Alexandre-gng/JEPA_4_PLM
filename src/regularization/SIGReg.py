"""
Sketched Isotropic Gaussian Regularization (SIGReg ou Strong SIGReg)

PROBLEME: Les architectures JEPA s'effondrent souvent, elles ap

=> L'objectif c'est de forcer la distribution des representations à suivre une distribution 
    isotropique gaussienne
=> Pour ça on calcule la "Empirical Characteristic Function" (ECF), sorte de transformée de
    Fourier de la distribution des representations, et on la compare à la "Characteristic 
    Function" (CF) d'une distribution isotropique gaussienne.

=> Objectif: minimiser la distance ECF - CF, pour forcer les représentations à suivre
    une distribution isotropique gaussienne.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class SIGRegLossConfig:
    sketch_dim: int = 64
    lambda_: float = 1.0


class SIGRegLoss(nn.Module):
    """
    SIGRegLoss implements the loss function for the Sketched Isotropic Gaussian Regularization (SIGReg) method, which is designed for self-supervised learning.
    The loss consists of a single component:
    - SIGReg Loss: Encourages the feature representations to follow an isotropic Gaussian distribution by minimizing the distance between the Empirical Characteristic Function (ECF) of the features and the Characteristic Function (CF) of a Gaussian distribution.
    
    See: LeJEPA: https://arxiv.org/abs/2305.17180
    """
    def __init__(self, config: SIGRegLossConfig):
        super().__init__()
        self.sketch_dim = config.sketch_dim
        self.lambda_ = config.lambda_
    

    def sigreg_strong_loss(self, x, sketch_dim=None):
        """
        Forces ECF(x) ~ ECF(Gaussian).
        Matches ALL Moments (Maximum Entropy Cloud).
        Exact implementation of LeJEPA Algorithm 1.

        From https://github.com/kreasof-ai/sigreg
        """
        N, C = x.size()

        # 1. Projection (The Observer)
        # Project channels down to sketch_dim
        if sketch_dim is None:
            sketch_dim = self.sketch_dim
        A = torch.randn(C, sketch_dim, device=x.device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

        # Integration points
        t = torch.linspace(-5, 5, 17, device=x.device)

        # Characteristic Function of Gaussian
        exp_f = torch.exp(-0.5 * t**2)

        # Empirical Characteristic Function
        # proj: [N, sketch_dim]
        proj = x @ A

        # args: [N, sketch_dim, T]
        args = proj.unsqueeze(2) * t.view(1, 1, -1)

        # ecf: [sketch_dim, T] (Mean over batch)
        ecf = torch.exp(1j * args).mean(dim=0)

        # Epps-Pulley test
        # Weighted L2 Distance: |ecf - gauss|^2 * gauss_weight
        # Compute the difference between the ECF and the excpected gaussian CF
        diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
        err = diff_sq * exp_f.unsqueeze(0)

        # Compute the integral using the trapezoidal rule
        loss = torch.trapz(err, t, dim=1) * N

        return loss.mean()
    

    def forward(self, z):
        """
        Compute the SIGReg loss for the given feature representations.

        Args:
            z: Tensor of shape (batch_size, feature_dim) representing the features from the model.
        Returns:
            loss: Scalar tensor representing the SIGReg loss.
        """
        if z.ndim > 2:
            z = z.reshape(-1, z.size(-1))
        
        loss = self.sigreg_strong_loss(z)
        return self.lambda_ * loss