import torch
import torch.nn as nn
import torch.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_channels=20, num_classes=1):
        super(EEGNet, self).__init__()
        
        # First Conv Layer
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Depthwise Conv Layer
        self.depthwiseConv = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.activation = nn.ELU()
        self.dropout1 = nn.Dropout(0.25)
        
        # Separable Conv Layer
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), groups=32, bias=False, padding=(0, 8)),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.25)
        )

        self.saved_features = None
        
        # Classification Layer
        self.fc1 = nn.Linear(480, num_classes)

    def forward(self, x, save_features=False):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout1(x)
        
        x = self.separableConv(x)
        x = F.avg_pool2d(x, (1, 8))
        
        x = x.view(x.size(0), -1)
        if save_features:
            self.saved_features = x
        x = F.sigmoid(self.fc1(x))
        
        if not save_features:
            return x
        else:
            return x, self.saved_features

if __name__ == "__main__":    
    model = EEGNet()
    input_tensor = torch.randn(100, 1, 20, 500)
    output = model(input_tensor)
    print(output.shape)

    # model.visualize_attention(0)