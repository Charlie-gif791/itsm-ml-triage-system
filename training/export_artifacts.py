import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def export_classifier_artifacts(
    model: nn.Module,
    label_map: Dict[str, int],
    best_epoch_metrics,
    output_dir: Path,
) -> None:
    """
    Export trained classifier artifacts for inference.

    Args:
        model:
            Trained classifier model.
        label_map:
            Mapping from label string -> class index.
        output_dir:
            Directory to write artifacts into.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Export model weights
    # --------------------------------------------------
    model_path = output_dir / "classifier.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Exported classifier to %s", model_path)

    # --------------------------------------------------
    # 2. Export label map
    # --------------------------------------------------
    label_map_path = output_dir / "label_map.json"
    if label_map_path.exists():
        logger.warning("Overwriting existing label_map.json")

    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    logger.info("Exported label map to %s", label_map_path)

    #--------------------------------------------------
    # 3. Save best metrics
    # --------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_epoch_metrics, f, indent=2)
    
    logger.info("Best epoch metrics: %s", best_epoch_metrics)
