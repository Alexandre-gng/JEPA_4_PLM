from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class VICRegLossConfig:
    mu: float = 1.0
    lambda_: float = 1.0
    nu: float = 1.0


class VICRegLoss(nn.Module):
    """
    VICRegLoss implements the loss function for the VICReg (Variance-Invariance-Covariance Regularization) method, which is designed for self-supervised learning. 
    The loss consists of three components:
    - Variance Loss: Encourages feature diversity by penalizing low variance in the feature representations.
    - Covariance Loss: Encourages feature decorrelation by penalizing high covariance between different feature dimensions.
    - Invariance Loss: Encourages feature invariance by penalizing the mean squared error between the features from two different views of the same data.

    Args:
        nu (float CONSTANT): Weight for the covariance loss component. Default is 1.0.
    
        mu (float): Weight for the variance loss component. Default is 1.0.
        lambda_ (float): Weight for the invariance loss component. Default is 1.0. lamda_ = mu > 1.0 is recommended
    """
    def __init__(self, mu=1.0, lambda_=1.0, nu=1.0):
        super(VICRegLoss, self).__init__()
        self.variance = 0.0
        self.covariance = 0.0
        self.invariance = 0.0
        self.mu = mu
        self.lambda_ = lambda_
        self.nu = nu
    
    def compute_variance(self, z, z_prime):
        """Compute the variance loss to encourage feature diversity
        Args:
            z: Tensor of shape (batch_size, feature_dim) representing the features from the first view.
            z_prime: Tensor of shape (batch_size, feature_dim) representing the features from the second view.
        """
        std_z = torch.sqrt(torch.var(z, dim=0) + 1e-04)
        std_z_prime = torch.sqrt(torch.var(z_prime, dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z)) + torch.mean(F.relu(1 - std_z_prime))
        self.variance = std_loss

    def compute_covariance(self, z, z_prime):
        """Compute the covariance loss to encourage feature decorrelation.
        Args:
            z: Tensor of shape (batch_size, feature_dim) representing the features from the first view.
            z_prime: Tensor of shape (batch_size, feature_dim) representing the features from the second view.
        """
        z = z - z.mean(dim=0)
        z_prime = z_prime - z_prime.mean(dim=0)
        cov_z = (z.T @ z) / (z.size(0) - 1)
        cov_z_prime = (z_prime.T @ z_prime) / (z_prime.size(0) - 1)
        # Compute the squared sum of the off-diagonal elements of the covariance matrices
        F.off_diagonal = lambda x: x - torch.diag(torch.diag(x))
        cov_loss = F.off_diagonal(cov_z).pow(2).sum() + F.off_diagonal(cov_z_prime).pow(2).sum()
        self.covariance = cov_loss

    def compute_invariance(self, z, z_prime):
        """Compute the invariance loss to encourage feature invariance
        Args:
            z: Tensor of shape (batch_size, feature_dim) representing the features from the first view.
            z_prime: Tensor of shape (batch_size, feature_dim) representing the features from the second view.
        """
        invariance_loss = F.mse_loss(z, z_prime)
        self.invariance = invariance_loss


    def forward(self, z, z_prime):
        if z.ndim > 2:
            z = z.reshape(-1, z.size(-1))
        if z_prime.ndim > 2:
            z_prime = z_prime.reshape(-1, z_prime.size(-1))

        self.compute_variance(z, z_prime)
        self.compute_covariance(z, z_prime)
        self.compute_invariance(z, z_prime)
        total_loss = (self.lambda_ * self.invariance) + (self.mu * self.variance) + (self.nu * self.covariance)

        return total_loss