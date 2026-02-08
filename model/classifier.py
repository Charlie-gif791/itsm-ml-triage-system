# model/classifier.py
import torch
import torch.nn as nn

class ITSMClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(
            encoder.config.hidden_size,
            num_classes,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits
