import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class ITSMDataset(Dataset):
    """
    PyTorch dataset for ITSM text classification.

    Returns items compatible with Hugging Face-style models:
        {
            "input_ids": Tensor,
            "attention_mask": Tensor,
            "labels": Tensor
        }
    """

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
