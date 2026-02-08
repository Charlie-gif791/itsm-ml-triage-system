import pandas as pd
import pathlib
import logging
from pathlib import Path
from data.schema import validate_schema
from utils.logging import setup_logging
from training.labels import build_label_mapping, encode_labels, save_label_mapping
from data.text import load_tokenizer, tokenize_texts

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    setup_logging()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Path to this file: .../itsm-ml-triage/data/ingest.py
THIS_FILE = Path(__file__).resolve()

# Repo root: .../itsm-ml-triage
REPO_ROOT = THIS_FILE.parents[1]

RAW_DATA_PATH = REPO_ROOT / "data" / "raw"

def load_raw_dataset(raw_path: Path):
    """
    Takes in the path to the raw dataset, and drops text columns that 
    are NaN, empty str, or whitespace-only.
    """

    # Load CSVs and validate
    issues = pd.read_csv(raw_path / "issues.csv")
    utterances = pd.read_csv(raw_path / "sample_utterances.csv")

    validate_schema(
        issues,
        required_columns=["id", "issue_type"],
        name="issues.csv",
    )

    validate_schema(
        utterances,
        required_columns=["issueid", "actionbody"],
        name="sample_utterances.csv",
    )

    # Drop invalid entries in utterances 
    missing_mask = (
        utterances["actionbody"].isna()
        | (utterances["actionbody"].str.strip() == "")
    )

    num_missing = missing_mask.sum()
    if num_missing > 0:
        logger.info(f"Dropping {num_missing} utterances with missing actionbody.")
    
    utterances = utterances.loc[~missing_mask].reset_index(drop=True)

    logger.debug("Loaded %d issues and %d utterances",
                 len(issues),
                 len(utterances),)
    
    LABELS_PATH = Path("artifacts/label_mapping.json")

    # Merge the two datasets
    merged = utterances.merge(
    issues[["id", "issue_type"]],
    left_on="issueid",
    right_on="id",
    how="inner"
    )

    # Rename columns for ease of use
    dataset = merged[["actionbody", "issue_type"]].rename(
    columns={
        "actionbody": "text",
        "issue_type": "label"
    }
    )

    # Tokenizer preview
    tokenizer = load_tokenizer()
    sample_encodings = tokenize_texts(
        dataset["text"].head(5),
        tokenizer,
    )

    logger.debug("Tokenizer output keys %s",
                 sample_encodings.keys())
    
    return dataset[["text", "label"]]
    




if __name__ == "__main__":
    logger.info("Resolved RAW_DATA_PATH: %s", RAW_DATA_PATH)
    logger.info(load_raw_dataset(RAW_DATA_PATH))
