1. Model Overview

- Brief description:
- Task: 3-class ticket classification
- Input: raw ticket text
- Output: class probabilities

2. Architecture

A pretrained MiniLM encoder generates contextual embeddings. The [CLS] representation is passed to a linear classification layer trained with cross-entropy loss.

- Encoder: sentence-transformers/all-MiniLM-L6-v2
- Hidden size
- Classification head structure
- Activation
- Loss function

3. Training Configuration

- Batch size
- Learning rate
- Epochs
- Max length
- Imbalance ratio
- Primary metric: macro F1

4. Data Handling

- Train/val split strategy
- Rebalancing method
- Label mapping persistence

4. Data Handling

- Train/val split strategy
- Rebalancing method
- Label mapping persistence

6. Known Limitations

- Performance sensitive to imbalance ratio
- Minority recall remains limited
- Small dataset size

7. Future Work

- Experiment with focal loss instead of sampling
- Try weighted cross-entropy
- Evaluate larger encoder backbone
- Perform stratified k-fold validation
- Explore threshold calibration for minority classes
