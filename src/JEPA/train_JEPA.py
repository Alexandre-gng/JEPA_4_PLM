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

    for full_sequences, masked_sequences, _ in data_loader:
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
        for full_sequences, masked_sequences, _ in data_loader:
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
    """Train the JEPA model for a specified number of epochs, evaluating on the validation set after each epoch.
    
    Args:
        jepa: The JEPA model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set (can be None to skip validation).
        loss_fn: The loss function to use for training.
        optimizer: The optimizer to use for training.
        device: The device to run the training on (e.g., 'cpu' or 'cuda').
        num_epochs: The number of epochs to train for.
    Returns:
        The trained JEPA model after the specified number of epochs.
    """

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



def test_1sequence(jepa: JEPA, full_sequence, masked_sequence, device: torch.device):
    """Run the trained JEPA model on a single sequence and return the context and target latent representations.
    
    Args:
        jepa: The trained JEPA model.
        full_sequence: The full protein sequence (string).
        masked_sequence: The masked version of the protein sequence (string).
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        A tuple containing the context and target latent representations for the input sequence
    """
    jepa.eval()
    with torch.no_grad():
        full_sequence = [full_sequence]
        masked_sequence = [masked_sequence]
        context_latent, target_latent = jepa(full_sequence, masked_sequence)
        return context_latent.cpu(), target_latent.cpu()



def test_jepa(jepa: JEPA, test_loader, device: torch.device):
    """
    Run the trained JEPA model on the test set and return the context and target latent representations.
    
    Args:
        jepa: The trained JEPA model.
        test_loader: DataLoader for the test set.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        A list of tuples containing the context and target latent representations for each batch in the test set
    """
    jepa.eval()
    results = []
    with torch.no_grad():
        for full_sequences, masked_sequences, _ in test_loader:
            full_sequences = list(full_sequences)
            masked_sequences = list(masked_sequences)
            context_latent, target_latent = jepa(full_sequences, masked_sequences)
            results.append((context_latent.cpu(), target_latent.cpu()))
    return results