import torch
from torch import optim
from typing import Any

from JEPA.JEPA import JEPA


def compute_loss(loss_fn: Any, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if hasattr(loss_fn, 'forward'):
        return loss_fn.forward(predicted, target)
    return loss_fn(predicted, target)


def train_epoch(
    jepa: JEPA,
    data_loader,
    optimizer: optim.Optimizer,
    loss_fn: Any,
    device: torch.device,
) -> float:
    jepa.train()
    total_loss = 0.0
    total_items = 0

    for full_sequences, masked_sequences in data_loader:
        full_sequences = list(full_sequences)
        masked_sequences = list(masked_sequences)

        context_latent, target_latent = jepa(full_sequences, masked_sequences)
        loss = compute_loss(loss_fn, context_latent, target_latent)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()

        batch_size = len(full_sequences)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(1, total_items)


def validate_epoch(jepa: JEPA, data_loader, loss_fn: Any, device: torch.device) -> float:
    jepa.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for full_sequences, masked_sequences in data_loader:
            full_sequences = list(full_sequences)
            masked_sequences = list(masked_sequences)
            context_latent, target_latent = jepa(full_sequences, masked_sequences)
            loss = compute_loss(loss_fn, context_latent, target_latent)
            batch_size = len(full_sequences)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / max(1, total_items)


def train_jepa(
    jepa: JEPA,
    train_loader,
    val_loader,
    loss_fn: Any,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
):
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(jepa, train_loader, optimizer, loss_fn, device)
        val_loss = None
        if val_loader is not None:
            val_loss = validate_epoch(jepa, val_loader, loss_fn, device)

        if val_loss is not None:
            print(f'Epoch {epoch}/{num_epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f}')
        else:
            print(f'Epoch {epoch}/{num_epochs} - train_loss={train_loss:.6f}')

    return jepa

