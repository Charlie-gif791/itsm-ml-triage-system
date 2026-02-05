# model/load.py

import logging
from typing import Optional

import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)


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
