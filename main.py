import logging
from pathlib import Path

from config import MODEL_PATH, PROJECT_ROOT, ARTIFACT_DIR, REBALANCE_TRAINING
from data.ingest import load_raw_dataset
from data.splits import make_train_val_splits
from model.load import load_encoder

from training.train_classifier import train_classifier
from training.export_artifacts import export_classifier_artifacts
from training.data_utils import rebalance_training_data
from training.evaluate import evaluate_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust the mode for partial or complete training
def maybe_downsample(df, mode: str):
        if mode == "smoke":
            return df.sample(n=min(200, len(df)), random_state=42)
        elif mode == "dev":
            return df.sample(frac=0.1, random_state=42)
        return df  # full

def main():
    # --------------------------------------------------
    # 1. Load and preprocess data
    # --------------------------------------------------
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

    logger.info("Loading and preprocessing dataset")

    data_df = load_raw_dataset(DATA_DIR)

    train_df, val_df = make_train_val_splits(
        df=data_df,
        text_col="text",
        label_col="label",
    )

    # --------------------------------------------------
    # 2. Load encoder
    # --------------------------------------------------
    logger.info("Loading encoder")

    encoder = load_encoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        freeze=False,
    )

    # --------------------------------------------------
    # 3. Train classifier
    # --------------------------------------------------
    logger.info("Training classifier")

    config = {
        "mode": "full", # "smoke" | "dev"| "full"
        "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 16, # 16 for "full"
        "learning_rate": 3e-5,
        "num_epochs": 8,
        "max_length": 256, # 256 for "full"
    }

    # Mode determines size of datasets
    train_df = maybe_downsample(train_df, config["mode"])
    val_df = maybe_downsample(val_df, config["mode"])

    # Rebalance training examples if set to true
    if REBALANCE_TRAINING:
        train_df = rebalance_training_data(train_df)

    if config["mode"] == "smoke":
        config["num_epochs"] = 1
        config["max_length"] = 64

    model, label_map, history = train_classifier(
        train_df=train_df,
        val_df=val_df,
        encoder=encoder,
        config=config,
    )

    # --------------------------------------------------
    # 4. Export artifacts
    # --------------------------------------------------
    logger.info("Exporting artifacts %s", ARTIFACT_DIR)

    best_epoch_metrics = max(history, key=lambda x: x["val_macro_f1"])

    export_classifier_artifacts(
        model=model,
        label_map=label_map,
        best_epoch_metrics=best_epoch_metrics,
        output_dir=PROJECT_ROOT / ARTIFACT_DIR,
    )


if __name__ == "__main__":
    main()
