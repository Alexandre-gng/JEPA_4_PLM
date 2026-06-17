import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from JEPA.JEPA import JEPA
from JEPA.train_JEPA import train_jepa

from regularization.VICReg import VICRegLoss, VICRegLossConfig
from regularization.SIGReg import SIGRegLoss, SIGRegLossConfig

from dataset.ProteinDataset import ProteinDataset
from main import TRAIN_PATH, VAL_PATH


print("Loading datasets and saving them as .pt files...")
train_dataset = ProteinDataset(root_path=TRAIN_PATH, masked_ratio=0.15, n_sequences=10000)
val_dataset = ProteinDataset(root_path=VAL_PATH, masked_ratio=0.15, n_sequences=200)

print(f"Train dataset loaded with {len(train_dataset)} sequences.")
print(f"Validation dataset loaded with {len(val_dataset)} sequences.")
# save the datasets as .pt files
torch.save(train_dataset, 'dataset/pt_10k/train_dataset.pt')
torch.save(val_dataset, 'dataset/pt_10k/val_dataset.pt')