# ITSM ML Triage System

## Overview

A production-structured machine learning system that classifies IT service tickets into operational categories with confidence-based abstention and API-based serving. Designed to mirror real-world ML systems, and emphasizes evaluation rigor under imbalance.

## Problem & Challenges

The dataset is heavily imbalanced toward the “Ticket” class, creating a realistic operational ML challenge. Early experiments showed high accuracy but poor minority-class performance, revealing majority-class collapse.

- Initial runs achieved ~94% accuracy
- Macro F1 ~0.32 exposed majority-class collapse
- Confusion matrix showed near-zero minority predictions
- Rebalancing strategy implemented (imbalance_ratio = 3)
- Macro F1 selected as primary evaluation metric

## Architecture Overview

### The System Is Structured Into Cleanly Separated Layers:

training/ – Model optimization and evaluation
model/ – Encoder and classifier definitions (pure ML logic)
policy/ – Business rules (confidence thresholds, abstention)
service/ – FastAPI inference layer
artifacts/ – Exported model weights and label maps
config.py – Centralized artifact and runtime configuration

Model inference and business policy are intentionally separated to mirror production ML system design.

### Features

- End-to-end ML lifecycle (ingest → split → train → evaluate → export → serve)
- Deterministic label mapping artifact
- Confidence-based abstention policy
- FastAPI inference service with Swagger UI
- Smoke/dev/full training modes for reproducibility

### Design Decisions

- Label map persisted as artifact to guarantee deterministic inference
- Model and policy layers separated to decouple statistical prediction from business rules
- Config centralized to avoid hardcoded paths and ensure portability
- Training modes implemented to balance iteration speed and full-model evaluation

## Modeling Approach

The system uses a lightweight transformer encoder with a classification head and emphasizes disciplined experimentation under class imbalance.

- Base encoder: all-MiniLM-L6-v2
- Transformer encoder with linear classification head
- Cross-entropy loss
- Rebalancing via capped majority sampling
- Dev vs full mode experimentation
- Learning rate sweep (3e-5 vs 4e-5 vs 5e-5)
- Selection criteria (macro F1 stability)

## Final Results

Full-mode training metrics:

- Validation Accuracy: 0.857
- Validation Macro F1: 0.424

Accuracy alone is misleading due to class imbalance. Macro F1 is the primary evaluation metric. Minority-class detection improved substantially compared to initial collapsed baseline (~0 F1 for minority classes). Minority class F1 scores improved from ~0 in early runs to meaningful detection after rebalancing.

## What I Would Improve Next

- Experiment with focal loss instead of sampling
- Try weighted cross-entropy
- Evaluate larger encoder backbone
- Perform stratified k-fold validation
- Explore threshold calibration for minority classes

## Running The Project

### Train The Model
```bash
python main.py --mode smoke
python main.py --mode dev
python main.py --mode full
```

### Run Tests
```bash
python -m tests.test_dataset
```

### Running The Inference API
```bash
python -m uvicorn service.app:app --reload
```
Visit:
```bash
http://localhost:8000/docs
```
Submit:
```bash
{
  "text": "User cannot access VPN"
}
```
Example response:
```bash
{
  "predicted_label": "HD Service",
  "confidence": 0.82,
  "abstained": false
}
```

## Repository Structure

- data/        Dataset loading and label handling
- model/       Model definitions and loading utilities
- training/    Training loop and optimization logic
- artifacts/   Generated training artifacts (label maps, weights)
- tests/       Sanity checks and unit tests

## Artifacts

Training generates the following artifacts:

- artifacts/label_map.json
- artifacts/classifier.pt

These are created during training and required for inference.
Pretrained artifacts are included for immediate inference.
To retrain the model from scratch:
```bash
python main.py --mode full
```
