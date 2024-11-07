import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Custom Dataset Class
class LipReadingDataset(Dataset):
    def __init__(self, data_dict, labels_dict, vocab=None, max_label_length=12):
        self.data_dict = data_dict
        self.labels_dict = labels_dict
        self.keys = list(data_dict.keys())
        self.max_label_length = max_label_length
        if vocab is None:
            self.vocab = self.build_vocab(labels_dict)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        frames = self.data_dict[key]  # Shape: (75, 48, 96, 3) -> (75 frames, 48 height, 96 width, 3 channels)

        frames = torch.tensor(frames, dtype=torch.float32)  # Convert to tensor
        frames = frames.permute(3,0, 1, 2)  # Change shape to (75, 3, 48, 96) - (frames, channels, height, width)
        print(f"Frames shape: {frames.shape}") 
        # Normalize to [0, 1]
        frames = frames / 255.0  

        label = self.labels_dict[key]
        label = " ".join(label)  # Convert list to string

        label_tokens = [self.vocab[word] for word in label.split()]
        label_tokens = label_tokens[:self.max_label_length]
        label_tokens += [self.vocab["<PAD>"]] * (self.max_label_length - len(label_tokens))

        return frames, torch.tensor(label_tokens, dtype=torch.long)

    def build_vocab(self, labels_dict):
        ''' Create a vocabulary based on the labels '''
        vocab = defaultdict(lambda: len(vocab))  # auto incrementing ID for each word
        vocab["<PAD>"]  # padding token
        for label in labels_dict.values():
            for word in label:
                vocab[word]
        return vocab

def get_dataloader(train_data_path, label_data_path, batch_size=2):
    train_data = np.load(train_data_path, allow_pickle=True).item()
    labels_dict = np.load(label_data_path, allow_pickle=True).item()
    dataset = LipReadingDataset(train_data, labels_dict, max_label_length=12)
    return DataLoader(dataset, batch_size=2, shuffle=True)

'''
train_data = np.load("dataset/traindata.npy", allow_pickle=True).item()
label_file_path = "dataset/alignments.npy"
labels_dict = np.load(label_file_path, allow_pickle=True).item()

dataset = LipReadingDataset(train_data, labels_dict)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
'''

'''Printing batch frames and labels to verify correct dataloader creation
for batch_frames, batch_labels in train_loader:
    print("Batch frames shape:", batch_frames.shape)  # Should be (batch_size, 75, 3, 48, 96)
    print("Batch labels (tokenized):", batch_labels)  # The tokenized labels for each batch item
    break  # Print only the first batch
'''