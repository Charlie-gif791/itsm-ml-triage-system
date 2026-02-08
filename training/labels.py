import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_label_mapping(labels):
    unique_labels = sorted(labels.unique())

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    logger.info(
        "Built label mapping with %d classes",
        len(label_to_id),
    )

    return label_to_id, id_to_label


def encode_labels(series, label_to_id):
    unknown = set(series.unique()) - set(label_to_id)

    if unknown:
        raise ValueError(f"Unknown labels encountered: {unknown}")

    return series.map(label_to_id)


def save_label_mapping(label_to_id, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(label_to_id, f, indent=2)

    logger.info("Saved label mapping to %s", path)


def load_label_mapping(path: Path):
    with path.open() as f:
        return json.load(f)
