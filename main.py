import logging
from pathlib import Path

from data.ingest import load_raw_dataset
from data.splits import make_train_val_splits
from model.load import load_encoder
from training.train_classifier import train_classifier
from training.export_artifacts import export_classifier_artifacts

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
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

    logger.info("Loading and preprocessing dataset")

    data_df = load_raw_dataset(
        DATA_DIR,
    )

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
        freeze=True,
    )

    # --------------------------------------------------
    # 3. Train classifier
    # --------------------------------------------------
    logger.info("Training classifier")

    config = {
        "mode": "smoke", # "smoke" | "dev"| "full"
        "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32, # 16 for "full"
        "learning_rate": 2e-5,
        "num_epochs": 5, # 5 for "full"
        "max_length": 128, # 256 for "full"
    }
    
    # The "mode" determines size of datasets
    train_df = maybe_downsample(train_df, config["mode"])
    val_df = maybe_downsample(val_df, config["mode"])

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
    artifacts_dir = PROJECT_ROOT / "artifacts"
    logger.info("Exporting artifacts %s", artifacts_dir)

    export_classifier_artifacts(
        model=model,
        label_map=label_map,
        output_dir=artifacts_dir,
    )


if __name__ == "__main__":
    main()
