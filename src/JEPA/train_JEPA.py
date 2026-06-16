import numpy as np

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
    list_context_latent, list_target_latent = [], []
    with torch.no_grad():
        for full_sequences, masked_sequences, _ in data_loader:
            full_sequences = list(full_sequences)
            masked_sequences = list(masked_sequences)
            context_latent, target_latent = jepa(full_sequences, masked_sequences)
            loss = compute_loss(loss_fn, context_latent, target_latent)
            batch_size = len(full_sequences)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            list_context_latent.append(context_latent)
            list_target_latent.append(target_latent)
    return total_loss / max(1, total_items), list_context_latent, list_target_latent



def get_latent_representations(jepa: JEPA, data_loader, device: torch.device):
    """
    Get the latent representations from the trained JEPA model for the given data loader.
    
    Args:
        jepa: The trained JEPA model.
        data_loader: DataLoader for the dataset to extract latent representations from.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        A list of tuples containing the context and target latent representations for each batch in the data loader.
    """
    jepa.eval()
    latent_representations = []

    with torch.no_grad():
        for full_sequences, masked_sequences, _ in data_loader:
            full_sequences = list(full_sequences)
            masked_sequences = list(masked_sequences)
            context_latent, target_latent = jepa(full_sequences, masked_sequences)
            latent_representations.append((context_latent.cpu(), target_latent.cpu()))

    return latent_representations



def get_mean_cosine_similarity(list_context_target_latent_pairs):
    """
    Compute the cosine similarity between context and target latents.
 
    Args:
        list_context_target_latent_pairs: list of (context_latent, target_latent)
            tensor pairs, each of shape (N_i, D). N_i can differ between pairs.
    Returns:
        (mean, std) of the cosine similarity pooled over every sample.
    """
    similarities = []
    for context_latent, target_latent in list_context_target_latent_pairs:
        context_latent = context_latent.detach()
        target_latent = target_latent.detach()
        similarity = torch.nn.functional.cosine_similarity(context_latent, target_latent, dim=1)
        similarities.append(similarity.cpu())
 
    all_similarities = torch.cat(similarities)
    return all_similarities.mean().item(), all_similarities.std().item()



def get_var_by_dim(list_latents):
    """
    Variance per dimension, number of dead dimensions, and effective rank (80% cumulative variance) over a pool of latents.
    Args:
        list_latents: list of latent tensors, any shape (..., D).
    Returns:
        (variances, dead_dims, effective_dim_80) — variances has shape (D,).
    """
    arrays = []
    for latent in list_latents:
        arr = latent.detach().cpu().numpy()
        arr = arr.reshape(-1, arr.shape[-1])  # flatten all but the feature dim
        arrays.append(arr)
    latents_np = np.concatenate(arrays, axis=0)  # (n_observations, D)

    variances = np.var(latents_np, axis=0)  # shape (D,), guaranteed
    dead_dims = int(np.sum(variances < 1e-4))

    X_centered = latents_np - latents_np.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    s_norm = s / s.sum()
    cumulative_variance = np.cumsum(s_norm)
    effective_dim_80 = int(np.argmax(cumulative_variance >= 0.80) + 1)

    return variances, dead_dims, effective_dim_80



def get_pairwise_cosine_similarity(latents, max_samples=2000):
    """
    Measures how close embeddings are to EACH OTHER in latent space — this is
    the real "collapse homogeneity" signal (it's the vectorized version of
    the protein-vs-protein cosine similarity loop you started with).

    Args:
        latents: list of tensors, any shape (..., D). Internally flattened to
            (n_observations, D), so it works whether each tensor is a single
            embedding (D,), a batch (N, D), or batched patches (N, n_patches, D).
        max_samples: random subsample cap. The pairwise matrix grows as O(n^2)
            in memory (e.g. 5000 samples -> 25M entries -> ~100MB in float32),
            so very large pools are subsampled for this specific computation.
    Returns:
        (mean, std) cosine similarity between DISTINCT embedding pairs.
        -> close to 1: embeddings are nearly identical (collapse).
        -> close to 0 (or negative): embeddings are spread out (healthy-ish).
    """
    arrays = []
    for latent in latents:
        arr = latent.detach().cpu()
        arr = arr.reshape(-1, arr.shape[-1])  # flatten all but the feature dim
        arrays.append(arr)
    pooled = torch.cat(arrays, dim=0)  # (N, D)

    if pooled.shape[0] > max_samples:
        idx = torch.randperm(pooled.shape[0])[:max_samples]
        pooled = pooled[idx]

    normed = torch.nn.functional.normalize(pooled, dim=1)
    sim_matrix = normed @ normed.T  # (n, n)

    n = sim_matrix.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = sim_matrix[off_diag_mask]

    return off_diag.mean().item(), off_diag.std().item()



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
            val_loss, list_context_latent, list_target_latent = validate_epoch(jepa, val_loader, loss_fn, device)
            cossim_pairwise, std_cossim_pairwise = get_pairwise_cosine_similarity(list_context_latent)
            cossim_targ_cont, std_cossim_targ_cont = get_mean_cosine_similarity(list(zip(list_context_latent, list_target_latent)))
            variances, dead_dims, effective_dim_80 = get_var_by_dim(list_context_latent)

        if val_loss is not None:
            print(f'Epoch {epoch}/{num_epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f} cosine_similarity between context and target latents={cossim_targ_cont:.6f} cosine_similarity between context latents={cossim_pairwise:.6f}')
            print(f'Std cosine similarity context/target ={std_cossim_targ_cont:.6f} Std cosine similarity context/context ={std_cossim_pairwise:.6f}')
            print(f'Mean variance by dimension: {variances.mean():.6f}')
            print(f'Dead dimensions: {dead_dims}')
            print(f'Effective dimension (80% variance): {effective_dim_80}')

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