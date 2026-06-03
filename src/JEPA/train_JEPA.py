import torch
from torch import nn as nn
from typing import Any

from JEPA.JEPA import JEPA


def compute_loss(loss_fn: Any, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute loss with any loss module or loss-like object.

    Supports:
    - nn.Module-based losses (VICRegLoss, nn.MSELoss, etc.)
    - custom classes implementing `forward`
    - callable loss functions
    """
    if hasattr(loss_fn, "forward"):
        return loss_fn.forward(predicted, target)
    return loss_fn(predicted, target)


def train_jepa(jepa: JEPA, full_sequence, masked_sequence):
    return jepa.forward(full_sequence, masked_sequence)





