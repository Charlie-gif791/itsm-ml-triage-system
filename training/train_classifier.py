from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

from data.dataset import ITSMDataset
from model.classifier import ITSMClassifier

logger = logging.getLogger(__name__)

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        loss, logits = model(**batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(dataloader)

def train_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    encoder,
    config: Dict,
) -> Tuple[nn.Module, Dict[str, int], List[Dict]]:
    """
    Train an ITSM classifier.

    Args:
        train_df, val_df:
            DataFrames with columns ["text", "label"]
            Labels are raw strings.
        encoder:
            Frozen transformer encoder.
        config:
            Training configuration dict.

    Returns:
        model:
            Trained classifier model.
        history:
            List of per-epoch metric dicts.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --------------------------------------------------
    # 1. Encode labels
    # --------------------------------------------------
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])

    num_classes = len(label_encoder.classes_)
    logger.info("Number of classes: %d", num_classes)

    label_map = {
        label: int(idx)
        for idx, label in enumerate(label_encoder.classes_)
    }

    # --------------------------------------------------
    # 2. Compute class weights
    # --------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # --------------------------------------------------
    # 3. Tokenize text
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )

    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )

    # --------------------------------------------------
    # 4. Build datasets and loaders
    # --------------------------------------------------
    train_dataset = ITSMDataset(train_encodings, y_train)
    val_dataset = ITSMDataset(val_encodings, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # macOS-safe
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # 5. Build model
    # --------------------------------------------------
    model = ITSMClassifier(
        encoder=encoder,
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    # --------------------------------------------------
    # 6. Training loop
    # --------------------------------------------------
    history: List[Dict] = []

    for epoch in range(config["num_epochs"]):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
            }
        )

        logger.info(
            "Epoch %d/%d - train loss: %.4f",
            epoch + 1,
            config["num_epochs"],
            train_loss,
        )

    return model, label_map, history
