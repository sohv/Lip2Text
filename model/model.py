import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import get_dataloader

class LipReadingModel(nn.Module):
    def __init__(self, in_channels=3, seq_length=75, num_classes=500):
        super(LipReadingModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Linear layer to project feature dimension to d_model (256)
        self.fc_proj = nn.Linear(128 * 48 * 96, 256)  # Assuming output shape of Conv3D: (batch, 128, 75, 48, 96)
        
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # First Convolution Layer
        x = self.conv1(x)
        x = F.relu(x)

        # Second Convolution Layer
        x = self.conv2(x)
        x = F.relu(x)

        batch, channels, frames, height, width = x.shape

        # Flatten frames for 2D convolution while keeping batch size intact
        x = x.view(batch * frames, channels, height, width)  # Flatten frames into one dimension

        # Flatten spatial dimensions (height, width) and project to d_model (256)
        x = x.view(batch * frames, -1)  # Flatten (batch * frames) x (channels * height * width)
        x = self.fc_proj(x)  # Project the flattened features to d_model

        # Reshape back to (batch, frames, feature_dim)
        x = x.view(batch, frames, 256)  # Now the feature dimension matches d_model (256)

        # Pass through transformer: (seq_len, batch, feature_dim)
        x = x.permute(1, 0, 2)  # Permute to shape: (seq_len, batch, feature_dim)

        # Apply transformer
        x = self.transformer(x, x)

        # Average over the sequence (frame dimension)
        x = x.mean(dim=0)  # Shape: (batch_size, feature_dim)

        # Fully connected layer to output the final result
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x

# Set the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipReadingModel()
model.to(device)

# Load data and labels using the get_dataloader function from dataloader.py
train_data_path = "dataset/traindata.npy"
label_data_path = "dataset/alignments.npy"
train_loader = get_dataloader(train_data_path, label_data_path, batch_size=2)

# Example criterion and optimizer
criterion = nn.CrossEntropyLoss()  # Assuming tokenized labels
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_frames, batch_labels in train_loader:
        # Send data to the device
        batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)

        # Ensure batch_labels is 1D (target shape should be [batch_size])
        batch_labels = batch_labels.squeeze()  # Flatten the labels if they have more than one dimension

        # Forward pass through the model
        output = model(batch_frames)  # Shape: (batch_size, num_classes)

        # Check if batch size is 2, which is the desired value
        assert output.size(0) == 2, f"Expected batch size 2, but got {output.size(0)}"

        # Ensure the output is of shape (batch_size, num_classes) for cross-entropy loss
        output = output.view(-1, output.size(-1))  # Flatten the output to (batch_size * num_classes)

        # Compute loss using tokenized labels
        loss = criterion(output, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero out the gradients
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model parameters
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
