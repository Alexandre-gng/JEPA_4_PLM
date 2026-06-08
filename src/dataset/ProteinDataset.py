import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """Custom Dataset for protein sequences and their masked versions."""

    def __init__(self, root_path: str, masked_ratio: float = 0.15, n_sequences: int | None = None):
        self.root_path = Path(root_path)
        self.masked_ratio = masked_ratio
        self.sequences = self._load_sequences(n_sequences)

    def _load_sequences(self, n_sequences: int | None):
        if self.root_path.is_file():
            df = pd.read_csv(self.root_path)
            sequences = df['text'].astype(str).tolist()
        elif self.root_path.is_dir():
            sequences = []
            csv_files = sorted(self.root_path.glob('*.csv'))
            for csv_file in csv_files:
                if n_sequences is not None and len(sequences) >= n_sequences:
                    break
                df = pd.read_csv(csv_file)
                sequences.extend(df['text'].astype(str).tolist())
            if n_sequences is not None:
                sequences = sequences[:n_sequences]
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.root_path}")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        masked_sequence = self.mask_sequence(sequence)
        return sequence, masked_sequence

    def mask_sequence(self, sequence: str, mask_ratio: float | None = None):
        if mask_ratio is None:
            mask_ratio = self.masked_ratio

        sequence_chars = list(sequence)
        if len(sequence_chars) == 0:
            return sequence

        num_to_mask = max(1, int(round(len(sequence_chars) * mask_ratio)))
        mask_indices = random.sample(range(len(sequence_chars)), num_to_mask)
        for idx in mask_indices:
            sequence_chars[idx] = 'MASK'
        return ''.join(sequence_chars)
