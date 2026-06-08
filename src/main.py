import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from JEPA.JEPA import JEPA
from JEPA.train_JEPA import train_jepa
from regularization.VICReg import VICRegLoss
from dataset.ProteinDataset import ProteinDataset

TRAIN_PATH = 'dataset/train'
VAL_PATH = 'dataset/validation'
TEST_PATH = 'dataset/test'
MODEL_SAVE_PATH = 'saved_models/jepa_model.pt'


def collate_sequences(batch):
    full_sequences, masked_sequences = zip(*batch)
    return list(full_sequences), list(masked_sequences)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = ProteinDataset(root_path=TRAIN_PATH, masked_ratio=0.15, n_sequences=10000)
    val_dataset = ProteinDataset(root_path=VAL_PATH, masked_ratio=0.15)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )

    model = JEPA(latent_dim=320, output_dim=320, tau=0.99).to(device)
    loss_fn = VICRegLoss(mu=1.0, lambda_=1.0, nu=1.0).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4,
    )

    print('JEPA model initialized and datasets loaded.')
    jepa = train_jepa(
        jepa=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=5,
    )
    
    os.makedirs(Path(MODEL_SAVE_PATH).parent, exist_ok=True)
    torch.save(jepa.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    main()



