# tests/test_dataset.py
import torch
import pandas as pd
from data.dataset import ITSMDataset

def test_itsm_dataset_uses_positional_indexing():
    labels = pd.Series([0, 1, 2], index=[10, 20, 30])
    encodings = {
        "input_ids": torch.zeros((3, 5), dtype=torch.long),
        "attention_mask": torch.ones((3, 5), dtype=torch.long),
    }

    ds = ITSMDataset(encodings, labels)
    item = ds[1]

    assert item["labels"].item() == 1

# Run test
test_itsm_dataset_uses_positional_indexing()