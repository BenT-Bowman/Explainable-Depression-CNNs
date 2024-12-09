import torch
import torch.nn as nn
import torch.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DeprNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 128, (1, 5)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(128, 64, (1, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(64, 64, (1, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(64, 32, (1, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(32, 32, (1, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1,2)),
            nn.Flatten(),
            nn.Linear(32*20*13, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return F.sigmoid(x)
    

class NeuromodulatedDeprNet(nn.Module):
    def __init__(self, num_channels=20, num_classes=2, dopamine_scale=1.0, serotonin_scale=1.0, max_dopamine=1.0, max_serotonin=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, (1, 5))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, (1, 5))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, (1, 5))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, (1, 3))
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, (1, 2))
        self.bn5 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d((1, 2))
        self.fc1 = nn.Linear(32 * num_channels * 13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        # Dopamine and serotonin scaling parameters
        self.dopamine_scale = dopamine_scale
        self.serotonin_scale = serotonin_scale
        self.max_dopamine = max_dopamine
        self.max_serotonin = max_serotonin

    def forward(self, x):
        # Convolutional Blocks
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)

        x = F.leaky_relu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Dopamine and Serotonin signals
        logits = self.fc1(x)
        softmax_probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        dopamine_signal = softmax_probs.max(dim=1, keepdim=True)[0]  # Confidence in top class
        serotonin_signal = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-9), dim=1, keepdim=True)  # Entropy

        # Clip dopamine and serotonin signals
        dopamine_signal = torch.clamp(dopamine_signal, 0, self.max_dopamine)
        serotonin_signal = torch.clamp(serotonin_signal, 0, self.max_serotonin)

        # Neuromodulated fully connected layers
        x = x * (1 + self.dopamine_scale * dopamine_signal) * (1 - self.serotonin_scale * serotonin_signal)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)  # Multi-class output

        return x
    
if __name__ == "__main__":
    model = NeuromodulatedDeprNet()

    tensor = torch.randn((32, 1, 20, 500))

    print(model(tensor).shape)