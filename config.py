# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = "artifacts"
MODEL_PATH = "artifacts/classifier.pt"
LABEL_MAP_PATH = "artifacts/label_map.json"
THRESHOLD = 0.6
DEVICE = "cpu"
REBALANCE_TRAINING = True
USE_CLASS_WEIGHTS = False