import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from EEGNET import EEGNet

class CAEW(nn.Module):
    def __init__(self, num_heads=10):
        super().__init__()
        # TODO: Create an automatic way of defining layer sizes
        self.t_conv1 = nn.Conv2d(1, 16, (1, 16), padding=(0, 8), bias=False)
        self.t_batchnorm1 = nn.BatchNorm2d(16)

        self.t_conv2 = nn.Conv2d(16, 32, (1, 64), padding=(0, 32), bias=False)
        self.t_batchnorm2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.25)
        self.activation = nn.ELU()

        self.t_batchnorm3 = nn.BatchNorm2d(1)
        self.dropout3 = nn.Dropout(0.25)

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(480, num_heads, batch_first=True), 2)
        self.t_conv3 = nn.Conv2d(32, 1, (1, 15), bias=False)

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.t_batchnorm1(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, (1,4))

        x = self.t_conv2(x)
        x = self.t_batchnorm2(x)
        x = self.dropout2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, (1,8))

        batch_size, filters, channels, seq_length = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch_size, channels, seq_length * filters)
        x = self.transformer(x)
        x = x.view(batch_size, channels, filters, seq_length).permute(0, 2, 1, 3)
        
        x = self.t_conv3(x)
        x = self.t_batchnorm3(x)
        x = self.dropout3(x)
        x = F.softmax(x, 2)
        return x

class CAEW_EEGNet(EEGNet):
    def __init__(self, num_heads=8):
        super().__init__()

        self.caew = CAEW(num_heads=num_heads)
        self.caew_weights = None
    
    def forward(self, x, save_features=False):
        self.caew_weights = self.caew(x)
        x = x*self.caew_weights

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
    from time import time
    model = CAEW_EEGNet()
    input_tensor = torch.randn(100, 1, 20, 500)
    t_0 = time()
    output = model(input_tensor)
    t_0 = time()-t_0
    print(output.shape, t_0)