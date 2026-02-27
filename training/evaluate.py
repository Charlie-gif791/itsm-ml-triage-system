from typing import Dict
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
) -> Dict[str, float]:
    """
    Evaluate classifier on validation data.
    """

    model.eval()

    val_total_loss = 0.0
    preds = []
    targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        val_loss = criterion(logits, labels)
        val_total_loss += val_loss.item()
        predictions = logits.argmax(dim=1)

        preds.extend(predictions.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    val_avg_loss = val_total_loss / len(dataloader)

    return {
        "val_loss": val_avg_loss,
        "val_accuracy": accuracy_score(targets, preds),
        "val_macro_f1": f1_score(targets, preds, average="macro"),
    }
