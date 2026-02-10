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
from training.labels import build_label_mapping, encode_labels
from training.evaluate import evaluate_classifier

logger = logging.getLogger(__name__)

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(logits, labels)
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
    label_to_id, id_to_label = build_label_mapping(train_df["label"])
    y_train = encode_labels(train_df["label"], label_to_id)
    y_val = encode_labels(val_df["label"], label_to_id)
    
    num_classes = len(label_to_id)
    logger.info("Number of classes: %d", num_classes)

    label_map = label_to_id

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
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
            criterion=criterion,
        )

        val_metrics = evaluate_classifier(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    criterion=criterion,
                )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["val_accuracy"],
                "val_macro_f1": val_metrics["val_macro_f1"],
            }
        )

        logger.info(
            "Epoch %d/%d - train loss: %.4f",
            epoch + 1,
            config["num_epochs"],
            train_loss,
            history,
        )

    return model, label_map, history
