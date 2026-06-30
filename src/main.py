import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from JEPA.JEPA import JEPA
from JEPA.train_JEPA import train_jepa

from regularization.VICReg import VICRegLoss, VICRegLossConfig
from regularization.SIGReg import SIGRegLoss, SIGRegLossConfig

from dataset.ProteinDataset import ProteinDataset


CSV_TRAIN_PATH = 'dataset/csv/train'
CSV_VAL_PATH = 'dataset/csv/validation'
CSV_TEST_PATH = 'dataset/csv/test'

PT_TRAIN_PATH = 'dataset/pt_10k/train_dataset.pt'
PT_VAL_PATH = 'dataset/pt_10k/val_dataset.pt'

MODEL_SAVE_PATH = 'saved_models/test_10.pt'


def collate_sequences(batch):
    full_sequences, masked_sequences, name = zip(*batch)
    return list(full_sequences), list(masked_sequences), list(name)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    
    # If a direct CSV import is needed
    """
    train_dataset = ProteinDataset(root_path=CSV_TRAIN_PATH, masked_ratio=0.15, n_sequences=)
    val_dataset = ProteinDataset(root_path=CSV_VAL_PATH, masked_ratio=0.15, n_sequences=3)
    """ 
    
    val_dataset = torch.load(PT_VAL_PATH, weights_only=False)
    print(f"Validation dataset loaded with {len(val_dataset)} sequences.")
    train_dataset = torch.load(PT_TRAIN_PATH, weights_only=False)
    print(f"Train dataset loaded with {len(train_dataset)} sequences.")
   
    
    print("Datasets loaded. Sample size - Train: {}, Validation: {}".format(len(train_dataset), len(val_dataset)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
        drop_last=False,
    )
    print(f"elements of train_loader: {len(train_loader)}")
    # print the number of keys in the first batch of the train_loader
    print(f"First batch keys: {train_loader.dataset[0]}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    
    print('DataLoaders created.')
    model = JEPA(latent_dim=160, output_dim=320, tau=0.0).to(device)
    print('JEPA model created with latent_dim=160, output_dim=320, tau=0.0.')
    loss_fn = SIGRegLoss(config=SIGRegLossConfig(lambda_=0.10, sketch_dim=320)).to(device)
    print('SIGRegLoss initialized with lambda_=0.10 and sketch_dim=320.')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
    )

    print('AdamW optimizer created for context encoder and predictor with learning rate 5e-4.')
    print('JEPA model initialized and datasets loaded.')
    jepa = train_jepa(
        jepa=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=50,
    )
    print('Training completed. Saving model...')
    os.makedirs(Path(MODEL_SAVE_PATH).parent, exist_ok=True)
    torch.save(jepa.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    main()