import torch
import torch.nn as nn
import torch.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, num_variables):
        super().__init__()
        self.mha = nn.MultiheadAttention(num_variables, 5, batch_first=True)
    def forward(self, x):
        shape = x.shape[0], x.shape[2], x.shape[3]
        x = x.view(shape)


        # x: (batch_size, num_variables, seq_len)
        
        # Conv1D layer
        x = x.permute(0, 2, 1)
        key, query, value = x,x,x
        x, _ = self.mha(query, key, value)
        x = x.permute(0, 2, 1)
        x = x.view(shape[0], 1, shape[1], shape[2])
        return x
    


class SelfAttention(nn.Module):
    def __init__(self, in_channels, k=8):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // k, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // k, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (in_channels // k) ** 0.5

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x.view(batch_size, channels, -1))  # B x C/k x N
        proj_key = self.key_conv(x.view(batch_size, channels, -1))  # B x C/k x N
        proj_value = self.value_conv(x.view(batch_size, channels, -1))  # B x C x N

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # B x N x N
        attention = F.softmax(energy * self.scale, dim=-1)  # Attention map

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, channels, width, height)  # B x C x W x H

        return out + x  # Residual connection



class EEGNet(nn.Module):
    def __init__(self, num_channels=20, num_classes=1, samples=128):
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
        
        # Classification Layer
        self.fc1 = nn.Linear(480, num_classes)

    def forward(self, x):
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
        x = F.sigmoid(self.fc1(x))
        
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ATTEEGNet(nn.Module):
    def __init__(self, num_channels=20, num_classes=1, samples=128, num_heads=8):
        super(ATTEEGNet, self).__init__()
        
        # First Conv Layer
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Depthwise Conv Layer
        self.depthwiseConv = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.activation = nn.ELU()
        self.dropout1 = nn.Dropout(0.25)
        
        # Multi-Head Attention Layer (Temporal)
        self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)

        # Separable Conv Layer
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), groups=32, bias=False, padding=(0, 8)),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.25)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout1(x)
        batch_size, channels, height, width = x.size()
        # Reshape to have width (sequence length) and height * channels (features)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, height * channels)  # Sequence: width

        # Apply MHA, setting embed_dim to height * channels in the MHA layer
        x, self.attn_weights = self.mha(x, x, x)
        

        # Reshape back to original format for the next layers
        x = x.view(batch_size, width, channels, height).permute(0, 2, 3, 1)
        # print(x.shape)

        x = F.avg_pool2d(x, (1, 4))
        # print(x.shape)
        x = self.separableConv(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))

        return x



if __name__ == "__main__":    
    model = ATTEEGNet()
    input_tensor = torch.randn(100, 1, 20, 500)
    output = model(input_tensor)
    print(output.shape)

    # model.visualize_attention(0)