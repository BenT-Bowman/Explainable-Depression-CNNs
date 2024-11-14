import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.custom_load_dataset import CWTDataset
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from modules.wavelet_CNN import CWT_CNN
from tqdm import tqdm


if __name__ == "__main__":
    folder_path = "cwt_data"
    dataset = CWTDataset(folder_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, persistent_workers=True)


    #TODO: WRITE CNN-ATT Model
    model = CWT_CNN()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    import torch.optim as optim

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            # print(images.shape)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            predictions = (outputs >= 0.5).float()  # Apply threshold to logits
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = correct_predictions / total_predictions

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")


