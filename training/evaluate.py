import torch
from sklearn.metrics import f1_score

def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            predictions = logits.argmax(dim=1)

            preds.extend(predictions.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return f1_score(targets, preds, average="macro")