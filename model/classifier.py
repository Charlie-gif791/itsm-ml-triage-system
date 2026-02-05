# model/classifier.py
import torch
import torch.nn as nn

class ITSMClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.encoder = encoder

        hidden_size = encoder.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_classes)

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS pooling
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits

        return logits
