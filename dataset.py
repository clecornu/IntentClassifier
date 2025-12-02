import torch
from torch.utils.data import random_split
import json
from preprocess import PreProcessData


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key:val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def prepare_data(max_length=64, seed=42, train_split = 0.8):
    pre = PreProcessData(max_length=max_length)
    texts, labels_to_idx, intent_response_dict, _ = pre.parse_json()
    text_encodings = pre.tokenize(texts)

    dataset = IntentDataset(text_encodings, labels_to_idx)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    num_labels = len(set(labels_to_idx))
    return train_dataset, val_dataset, intent_response_dict, pre, num_labels



