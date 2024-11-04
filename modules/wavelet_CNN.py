import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CWT_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(246_016, 128)
        self.fc2 = nn.Linear(128, 1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(1,2), stride=(1, 2))
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(1,2), stride=(1, 2))
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(1,2), stride=(1, 2))
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return x
    
if __name__ == "__main__":
    t = torch.randn(32, 1, 31, 500)
    print(t.shape)
    model = CWT_CNN()
    print(model(t).shape)