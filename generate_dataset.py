import numpy as np
import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH = "/Users/sohan/Desktop/Lip2Text/processed_data" 
ALIGN_PATH = "/Users/sohan/Desktop/Lip2Text/align"
N_CLASSES = 500  # Number of unique word labels
BATCH_SIZE = 32  # Batch size for data loaders
SEQ_LENGTH = 29  # Sequence length of lip frames

# 1. Load .npy files containing lip regions
# Load .npy files containing lip regions
def load_lip_data(data_path):
    npy_files = glob.glob(os.path.join(data_path, "*.npy"))
    data = {os.path.basename(f).replace("_lips.npy", ""): np.load(f) for f in npy_files}  # Adjust this line
    return data

# 2. Load and parse .align files for word labels
def load_alignments(align_path):
    align_files = glob.glob(os.path.join(align_path, "*.align"))
    labels = {}
    for file in align_files:
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    _, _, word = parts
                    video_id = os.path.basename(file).replace(".align", "")
                    labels[video_id] = word
    return labels

# 3. Prepare data splits
def prepare_splits(data, labels, test_size=0.2, val_size=0.1):
    keys = list(data.keys())
    if len(keys) == 0:
        raise ValueError("No keys available for splitting the dataset.")
    
    random.shuffle(keys)
    train_keys, test_keys = train_test_split(keys, test_size=test_size)
    train_keys, val_keys = train_test_split(train_keys, test_size=val_size / (1 - test_size))

    return train_keys, val_keys, test_keys


# 4. Define to_categorical class
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# 4. Create a Dataset class
class LipReadingDataset(Dataset):
    def __init__(self, data, labels, keys):
        self.data = data
        self.labels = labels
        self.keys = keys
        self.word_to_index = {word: idx for idx, word in enumerate(set(labels.values()))}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        X = self.data[key]
        y = self.labels_to_categorical(self.labels[key])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def labels_to_categorical(self, label):
        return to_categorical(self.word_to_index[label], num_classes=N_CLASSES)

# Main Execution: Loading Data, Labels, and Generators
if __name__ == "__main__":
    # Load data and labels
    lip_data = load_lip_data(DATA_PATH)
    align_labels = load_alignments(ALIGN_PATH)

    # Debugging output
    print(f"Loaded lip data: {len(lip_data)} entries")
    print(f"Loaded align labels: {len(align_labels)} entries")

    # Filter data and labels to match available keys
    common_keys = list(set(lip_data.keys()) & set(align_labels.keys()))
    print(f"Common keys: {len(common_keys)} entries")

    lip_data = {k: lip_data[k] for k in common_keys}
    align_labels = {k: align_labels[k] for k in common_keys}

    # Prepare splits
    train_keys, val_keys, test_keys = prepare_splits(lip_data, align_labels)

    # Define the directory for saving splits
    SPLIT_DIR = "dataset"  # Change this to your desired output directory

    # Create the output directory if it doesn't exist
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # Save the training, validation, and test splits
    np.save(os.path.join(SPLIT_DIR, "train_keys.npy"), train_keys)
    np.save(os.path.join(SPLIT_DIR, "val_keys.npy"), val_keys)
    np.save(os.path.join(SPLIT_DIR, "test_keys.npy"), test_keys)

    # Save the corresponding lip data and labels for each split
    for split_name, keys in zip(['train', 'val', 'test'], [train_keys, val_keys, test_keys]):
        split_data = {k: lip_data[k] for k in keys}
        split_labels = {k: align_labels[k] for k in keys}
        
        np.save(os.path.join(SPLIT_DIR, f"{split_name}_data.npy"), split_data)
        np.save(os.path.join(SPLIT_DIR, f"{split_name}_labels.npy"), split_labels)

    # Create Dataset and DataLoader for PyTorch
    train_dataset = LipReadingDataset(lip_data, align_labels, train_keys)
    val_dataset = LipReadingDataset(lip_data, align_labels, val_keys)
    test_dataset = LipReadingDataset(lip_data, align_labels, test_keys)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Example: Verify shapes of generated batches
    X_batch, y_batch = next(iter(train_loader))
    print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

    print("Training, validation, and test sets saved successfully.")
