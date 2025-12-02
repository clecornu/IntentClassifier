import os

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model import TransformerModel
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


class TrainModel:
    def __init__(self, num_epochs):
        train_dataset, val_dataset, intent_response_dict, pre, num_labels = dataset.prepare_data()
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

        self.num_epochs = num_epochs
        self.num_labels = num_labels
        self.model = TransformerModel(num_labels)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 5e-5)

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0
            for batch in self.train_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                self.optimizer.zero_grad()

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {avg_loss:.4f}")
        torch.save(self.model.state_dict(), "intent_model.pt")

    def val_model(self):
        self.model.eval()
        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad() :
            for batch in self.val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct/total
        print(f"Evaluation {avg_loss:.4f}, Accuracy = {accuracy}")

train = TrainModel(5)
train.train_model()
train.val_model()






