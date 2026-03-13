# ITSM ML Triage System

A production-structured machine learning system that classifies IT service tickets into operational categories. The system applies confidence-based abstention and serves predictions through a REST API, with an emphasis on evaluation rigor under class imbalance.

**Live demo:** [https://itsm-ml-triage-system.onrender.com/docs](https://itsm-ml-triage-system.onrender.com/docs)

---

## Problem & Motivation

IT service ticket datasets are often heavily skewed toward a dominant class — in this case, "Ticket" — creating a realistic and challenging operational ML problem. A naive model achieves high accuracy while failing silently on minority classes.

Initial experiments exposed this failure mode directly:

- Accuracy of ~94% masked near-zero minority-class predictions
- Macro F1 of ~0.32 revealed majority-class collapse
- Confusion matrix confirmed the model was essentially ignoring minority classes

Addressing this required rethinking the evaluation criteria and rebalancing strategy before any architectural changes.

---

## Modeling Approach

The system uses a lightweight transformer encoder with a linear classification head, fine-tuned with a rebalancing strategy to counteract class imbalance.

| Component | Detail |
|---|---|
| Base encoder | `all-MiniLM-L6-v2` |
| Classification head | Linear layer over encoder output |
| Loss function | Cross-entropy |
| Rebalancing | Capped majority sampling (`imbalance_ratio = 3`) |
| Primary metric | Macro F1 (not accuracy) |
| LR sweep | 3e-5, 4e-5, 5e-5 |
| Selection criterion | Macro F1 stability across runs |

---

## Results

Full-mode training metrics:

| Metric | Value |
|---|---|
| Validation Accuracy | 0.857 |
| Validation Macro F1 | 0.424 |

Accuracy is not the primary signal here — Macro F1 is. Minority-class F1 improved from ~0 in the initial collapsed baseline to meaningful detection after rebalancing.

---

## Architecture

The system is organized into cleanly separated layers, designed to mirror production ML systems:

```
training/    Model optimization and evaluation
model/       Encoder and classifier definitions (pure ML logic)
policy/      Business rules: confidence thresholds and abstention logic
service/     FastAPI inference layer
artifacts/   Exported model weights and label maps
config.py    Centralized artifact and runtime configuration
```

Model inference and business policy are intentionally separated to decouple statistical prediction from operational rules.

### Key Design Decisions

- **Label map persisted as artifact** — guarantees deterministic inference across environments
- **Policy layer isolated from model layer** — business rules can change without touching ML logic
- **Centralized config** — eliminates hardcoded paths and improves portability
- **Training modes (smoke / dev / full)** — balances iteration speed with full evaluation fidelity

---

## Running the Project

### Train the Model

```bash
python main.py --mode smoke   # Fast sanity check
python main.py --mode dev     # Reduced dataset, quick iteration
python main.py --mode full    # Full training run
```

### Run Tests

```bash
python -m tests.test_dataset
```

### Start the Inference API

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the Swagger UI.

**Example request:**
```json
POST /predict
{
  "text": "User cannot access VPN"
}
```

**Example response:**
```json
{
  "predicted_label": "HD Service",
  "confidence": 0.82,
  "abstained": false
}
```

---

## Deployment

The inference service is deployed on Render as a FastAPI application. It loads trained model artifacts at startup to avoid repeated initialization and ensure low-latency inference.

**Production start command:**
```bash
uvicorn service.app:app --host 0.0.0.0 --port 10000
```

---

## Artifacts

Training produces two artifacts required for inference:

| Artifact | Purpose |
|---|---|
| `artifacts/label_map.json` | Deterministic label mapping |
| `artifacts/classifier.pt` | Trained classifier weights |

Pretrained artifacts are included for immediate inference. To retrain from scratch:

```bash
python main.py --mode full
```

---

## Repository Structure

```
data/        Dataset loading and label handling
model/       Model definitions and loading utilities
training/    Training loop and optimization logic
artifacts/   Generated artifacts (label maps, weights)
tests/       Sanity checks and unit tests
```

---

## What I Would Improve Next

- Experiment with focal loss as an alternative to resampling
- Evaluate weighted cross-entropy
- Test a larger encoder backbone
- Perform stratified k-fold validation
- Explore per-class threshold calibration for minority classes

---

## Dataset

This project uses the publicly available Help Desk Tickets dataset:

> Abdellatif, Mohammad (2025). "Help Desk Tickets." Mendeley Data, V1. [doi: 10.17632/btm76zndnt.1](https://doi.org/10.17632/btm76zndnt.1)

The dataset consists of labeled IT service requests with significant class imbalance, which directly motivates the modeling approach used here.
