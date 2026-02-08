## Project Status

✔ End-to-end training pipeline implemented  
✔ Label map generation and artifact export stabilized  
✔ Model training modes: smoke / dev / full  
✔ Inference bundle loading implemented  
✔ Dataset and training logic sanity-checked with tests  

Next phase: inference API + evaluation metrics

## Running the Project

### Train the model
```bash
python main.py --mode smoke
python main.py --mode dev
python main.py --mode full
```

### Run Tests
```bash
python -m tests.test_dataset
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
