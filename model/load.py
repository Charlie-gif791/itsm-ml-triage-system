# model/load.py
import logging
from typing import Optional, Dict

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

import json
from pathlib import Path

from model.classifier import ITSMClassifier

logger = logging.getLogger(__name__)

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def predict(text: str, bundle: Dict) -> Dict:
    """
    Run inference on a single text input.

    Returns:
        {
            "predicted_label": str,
            "confidence": float,
            "logits": List[float]
        }
    """
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    label_map = bundle["label_map"]
    device = bundle["device"]

    # -----------------------------
    # Tokenize
    # -----------------------------
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

    # -----------------------------
    # Post-process
    # -----------------------------
    probs = F.softmax(logits, dim=-1)
    confidence, pred_idx = torch.max(probs, dim=-1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()

    return {
        "predicted_label": label_map[pred_idx],
        "confidence": confidence,
        "logits": logits.squeeze(0).tolist(),
    }

def load_inference_bundle(
        label_map_path: Path,
        device: str = "cpu", 
    ) -> Dict:
    """
    Load all artifacts required for inference.

    Returns a dict containing:
      - model: torch.nn.Module
      - tokenizer: Hugging Face tokenizer
      - label_map: Dict[int, str]
      - device: torch.device
    """
    device = torch.device(device)

    # -----------------------------
    # Load label map
    # -----------------------------
    if not label_map_path.exists():
        raise FileNotFoundError(
            "label_map.json not found. "
            "Train the model before running inference."
        )

    with open(label_map_path, "r") as f:
        label_map_str = json.load(f)

    # Invert to index -> label
    label_map = {int(v): k for k, v in label_map_str.items()}

    num_classes = len(label_map)

    # -----------------------------
    # Load encoder + tokenizer
    # -----------------------------
    encoder = AutoModel.from_pretrained(ENCODER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    for param in encoder.parameters():
        param.requires_grad = False

    # -----------------------------
    # Build classifier
    # -----------------------------
    model = ITSMClassifier(
        encoder=encoder,
        num_classes=num_classes,
    )

    # -----------------------------
    # Load trained weights
    # -----------------------------
    state_dict = torch.load(
        PROJECT_ROOT / "artifacts" / "classifier.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_map": label_map,
        "device": device,
    }

def load_encoder(
    model_name: str,
    freeze: bool = True,
    device: Optional[torch.device] = None,
):
    """
    Load a pretrained transformer encoder.

    Args:
        model_name:
            Hugging Face model identifier.
        freeze:
            Whether to freeze encoder parameters.
        device:
            Torch device. If None, inferred automatically.

    Returns:
        encoder:
            Transformer model producing hidden states.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading encoder: %s", model_name)

    encoder = AutoModel.from_pretrained(model_name)
    encoder.to(device)
    encoder.eval()

    if freeze:
        logger.info("Freezing encoder parameters")
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder
