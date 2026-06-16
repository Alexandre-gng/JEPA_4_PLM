import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MAX_SEQ_LENGTH = 1024  # Define a maximum sequence length for padding/truncation if needed
MIN_SEQ_LENGTH = 20    # Define a minimum sequence length to filter out very short sequences

class ProteinDataset(Dataset):
    """Custom Dataset for protein sequences and their masked versions."""

    def __init__(self, root_path: str, masked_ratio: float = 0.15, n_sequences: int | None = None):
        self.root_path = Path(root_path)
        self.masked_ratio = masked_ratio
        # load tokenizer once so we can use its `mask_token` when creating masked sequences
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.mask_token = self.tokenizer.mask_token or "<mask>"
        except Exception:
            # fallback to a generic mask token if tokenizer unavailable in this environment
            self.tokenizer = None
            self.mask_token = "<mask>"
        self.sequences, self.prot_id = self._load_sequences(n_sequences)
        self.masked_sequences = [self.mask_sequence(seq, self.masked_ratio) for seq in self.sequences]


    def _load_sequences(self, n_sequences: int | None, max_seq_length: int = MAX_SEQ_LENGTH) -> tuple[list[str], list[str]]:
        """Load sequences from a CSV file or all CSV files in a directory, with optional limit on number of sequences."""
        if self.root_path.is_file():
            df = pd.read_csv(self.root_path)
            df = df[df['text'].str.len() <= max_seq_length]  # Filter sequences longer than max_seq_length
            df = df[df['text'].str.len() >= MIN_SEQ_LENGTH]  # Filter sequences shorter than min_seq_length
            sequences = df['text'].astype(str).tolist()
            prot_id = df["name"].astype(str).tolist()
        elif self.root_path.is_dir():
            sequences = []
            prot_id = []
            csv_files = sorted(self.root_path.glob('*.csv'))
            for csv_file in csv_files:
                if n_sequences is not None and len(sequences) >= n_sequences:
                    break
                df = pd.read_csv(csv_file)
                df = df[df['text'].str.len() <= max_seq_length]  # Filter sequences longer than max_seq_length
                df = df[df['text'].str.len() >= MIN_SEQ_LENGTH]  # Filter sequences shorter than min_seq_length
                sequences.extend(df['text'].astype(str).tolist())
                prot_id.extend(df["name"].astype(str).tolist())
            if n_sequences is not None:
                sequences = sequences[:n_sequences]
                prot_id = prot_id[:n_sequences]
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.root_path}")
        return sequences, prot_id
    
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        masked_sequence = self.masked_sequences[idx]
        prot_id = self.prot_id[idx]
        return sequence, masked_sequence, prot_id

    def mask_sequence(self, sequence: str, mask_ratio: float = 0.15) -> str:
        sequence_chars = list(sequence)
        if len(sequence_chars) == 0:
            return sequence

        num_to_mask = max(1, int(round(len(sequence_chars) * mask_ratio)))
        mask_indices = random.sample(range(len(sequence_chars)), num_to_mask)
        for idx in mask_indices:
            # replace single residue by the tokenizer's mask token (string form)
            sequence_chars[idx] = self.mask_token
        return ''.join(sequence_chars)


    @staticmethod
    def mask_sequence_static(sequence: str, mask_ratio: float = 0.15) -> str:
        """Backward-compatible static helper that uses a freshly-loaded tokenizer
        (slower) to replace masked residues with the tokenizer's mask token.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            mask_token = tokenizer.mask_token or "<mask>"
        except Exception:
            mask_token = "<mask>"

        sequence_chars = list(sequence)
        if len(sequence_chars) == 0:
            return sequence

        num_to_mask = max(1, int(round(len(sequence_chars) * mask_ratio)))
        mask_indices = random.sample(range(len(sequence_chars)), num_to_mask)
        for idx in mask_indices:
            sequence_chars[idx] = mask_token
        return ''.join(sequence_chars)
