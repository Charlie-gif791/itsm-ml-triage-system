import logging
import transformers
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_tokenizer():
    logger.info("Loading tokenizer: %s", ENCODER_NAME)
    return AutoTokenizer.from_pretrained(ENCODER_NAME)


def tokenize_texts(texts, tokenizer, max_length=128):
    """
    Tokenize raw text into model-ready tensors.
    """
    encodings = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    logger.info(
        "Tokenized %d texts (max_length=%d)",
        len(texts),
        max_length,
    )

    return encodings
