import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class CWTDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the main folder containing class subfolders.
            transform (callable, optional): Optional transform to be applied to each sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.npz_files = []
        self.labels = []

        # Collect .npz file paths and their corresponding labels
        for class_label, class_name in enumerate(sorted(os.listdir(folder_path))):
            class_folder = os.path.join(folder_path, class_name)
            if os.path.isdir(class_folder):
                for file_name in os.listdir(class_folder):
                    if file_name.endswith('.npz'):
                        self.npz_files.append(os.path.join(class_folder, file_name))
                        self.labels.append(class_label)  # Assign class label based on folder

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        label = self.labels[idx]

        # Load the entire 3D array (all channels)
        with np.load(npz_path) as data:
            image = data['image'].astype(np.float32)  # Shape: (Height, Channels, Width)
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        # image = np.transpose(image, (1, 0, 2))  # Shape: (Channels, Height, Width)

        if self.transform:
            image = self.transform(image)

        # Convert the image and label to PyTorch tensors
        # image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)  # Label as a long tensor for classification

        return image, label

class ResizeMoreChannels:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Check if img is a tensor
        if isinstance(img, torch.Tensor):            
            # Resize using torchvision's functional interface
            img_resized = F.resize(img, self.size)
            return img_resized
        
        raise TypeError("Input should be a torch.Tensor")


if __name__ == "__main__":
    # Example usage
    folder_path = "cwt_data"
    channel_index = 0  # Select the EEG channel
    dataset = CWTDataset(folder_path)

    # Set up DataLoader for batching and shuffling
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example: Iterate through batches
    for images, labels in data_loader:
        print(images.shape, type(images))  # Each batch will have shape [batch_size, height, width]
        print(labels.shape, type(labels))  # Each batch will have shape [batch_size, height, width]
