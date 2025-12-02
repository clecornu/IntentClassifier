import torch
import torch.nn as nn
from transformers import DistilBertModel

class TransformerModel(nn.Module):
    def __init__(self, num_labels, dropout=0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, **batch):
        outputs = self.distilbert(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"]
        )

        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
