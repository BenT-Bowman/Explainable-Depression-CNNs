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

# class ATTEEGNet(nn.Module):
#     def __init__(self, num_channels=20, num_classes=1, samples=128, num_heads=4):
#         super(ATTEEGNet, self).__init__()
        
#         # First Conv Layer
#         self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
#         self.batchnorm1 = nn.BatchNorm2d(16)
        
#         # Depthwise Conv Layer
#         self.depthwiseConv = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
#         self.batchnorm2 = nn.BatchNorm2d(32)
#         self.activation = nn.ELU()
#         self.dropout1 = nn.Dropout(0.25)
        
#         # Multi-Head Attention Layer
#         self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)

#         # Separable Conv Layer
#         self.separableConv = nn.Sequential(
#             nn.Conv2d(32, 32, (1, 16), groups=32, bias=False, padding=(0, 8)),
#             nn.Conv2d(32, 32, (1, 1), bias=False),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Dropout(0.25)
#         )

#         self.flatten = nn.Flatten()
#         self.attn_weights = None
        
#         # Classification Layer
#         self.fc1 = nn.Linear(480, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batchnorm1(x)
#         x = self.depthwiseConv(x)
#         x = self.batchnorm2(x)
#         x = self.activation(x)
#         x = F.avg_pool2d(x, (1, 4))
#         x = self.dropout1(x)

#         # Reshape x for MHA
#         batch_size, channels, height, width = x.size()
#         x = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, 1, channels)  # (B*H*W, 1, C)

#         # Apply Multi-Head Attention
#         x, self.attn_weights = self.mha(x, x, x)  # Self-attention
#         x = x.view(batch_size, height, width, channels)  # Reshape back to (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # Permute back to (B, C, H, W)

#         x = self.separableConv(x)
#         x = F.avg_pool2d(x, (1, 8))
        
#         # x = x.view(x.size(0), -1)
#         x = self.flatten(x)
#         x = torch.sigmoid(self.fc1(x))
        
#         return x
#     def visualize_attention(self, head_idx=3):
#         """
#         Visualize the attention weights from a specific head.

#         Args:
#         - attn_weights: The attention weights tensor (shape: [B*H*W, num_heads, L, L])
#         - head_idx: The index of the attention head to visualize.
#         """
#         # Select the attention weights for the specified head
#         attn = self.attn_weights[:, head_idx].detach().cpu().numpy()  # [B*H*W, L, L]
#         print("Attention Weights Statistics:")
#         print(attn)
#         print(f"Min: {attn.min()}, Max: {attn.max()}, Mean: {attn.mean()}")
        
        # # Ensure the attention weights are 2D for visualization
        # if attn.ndim == 2:
        #     fig, ax = plt.subplots()
        #     cax = ax.matshow(attn, cmap='viridis')
        #     fig.colorbar(cax)

        #     plt.title(f'Attention Weights for Head {head_idx}')
        #     plt.xlabel('Key Position')
        #     plt.ylabel('Query Position')
        #     plt.show()
        # else:
        #     print(f"Expected 2D attention weights but got shape {attn.shape}")


# class ATTEEGNet(nn.Module):
#     def __init__(self, num_channels=20, num_classes=1, samples=128, num_heads=1):
#         super(ATTEEGNet, self).__init__()

#         # EEGNet layers
#         self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
#         self.batchnorm1 = nn.BatchNorm2d(16)
        
#         self.depthwiseConv = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
#         self.batchnorm2 = nn.BatchNorm2d(32)
#         self.activation = nn.ELU()
#         self.dropout1 = nn.Dropout(0.25)
        
#         self.separableConv = nn.Sequential(
#             nn.Conv2d(32, 32, (1, 16), groups=32, bias=False, padding=(0, 8)),
#             nn.Conv2d(32, 32, (1, 1), bias=False),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Dropout(0.25)
#         )
        
#         # Temporal Attention layer (assuming you have a custom implementation)
#         self.temporal_attention = TemporalAttention(in_features=250, time_steps=samples)
        
#         # Final classification layer
#         self.fc1 = nn.Linear(992, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batchnorm1(x)
#         x = self.depthwiseConv(x)
#         x = self.batchnorm2(x)
#         x = self.activation(x)
#         x = F.avg_pool2d(x, (1, 4))
#         x = self.dropout1(x)
        
#         x = self.temporal_attention(x).unsqueeze(2)
#         x = self.separableConv(x)
        
#         # # Apply temporal attention before pooling
        
#         x = F.avg_pool2d(x, (1, 8))
#         x = x.view(x.size(0), -1)
#         x = torch.sigmoid(self.fc1(x))
        
#         return x

# # Squeeze-and-Excitation (SE) Block for Channel Attention
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # Squeeze: Global pooling across the temporal dimension
#         b, c, t = x.size()  # [batch_size, channels, timesteps]
#         y = self.avg_pool(x).view(b, c)  # Shape: [batch_size, channels]

#         # Excitation: Channel-wise fully connected layers
#         y = self.fc(y).unsqueeze(-1)  # Shape: [batch_size, channels, 1]

#         # Scale: Re-scale the input tensor by the channel attention
#         return x * y.expand_as(x)  # Shape: [batch_size, channels, timesteps]

# # Example CNN with SE Block (Channel Attention)
# class ATTEEGNet(nn.Module):
#     def __init__(self, num_channels=20, num_classes=1):
#         super(ATTEEGNet, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
#         self.se_block = SEBlock(channels=64)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, padding=2)
        
#         # Channel Attention (SE) Block

#         # Fully connected layer
#         self.fc = nn.Linear(512*62, num_classes)  # Adjust depending on downsampling

#     def forward(self, x):
#         x = x.view(x.shape[0], x.shape[2], x.shape[3])
#         # Convolutional layers
#         x = F.relu(self.conv1(x))  # Shape: [batch_size, 32, num_timesteps]
#         x = F.max_pool1d(x, kernel_size=2)  # Downsample to (num_timesteps // 2)
        
#         x = F.relu(self.conv2(x))  # Shape: [batch_size, 64, num_timesteps // 2]
#         x = F.max_pool1d(x, kernel_size=2)  # Downsample to (num_timesteps // 4)

#         # Apply SE Block for Channel Attention
#         x = self.se_block(x)  # Apply channel attention

#         x = F.relu(self.conv3(x))  # Shape: [batch_size, 64, num_timesteps // 2]
#         x = F.max_pool1d(x, kernel_size=2)  # Downsample to (num_timesteps // 4)

#         x = F.relu(self.conv4(x))  # Shape: [batch_size, 64, num_timesteps // 2]
#         x = F.max_pool1d(x, kernel_size=2)  # Downsample to (num_timesteps // 4)

#         # # Flatten and fully connected layer for classification
#         x = x.view(x.size(0), -1)  # Flatten to [batch_size, channels * timesteps]
#         x = self.fc(x)

#         return x




if __name__ == "__main__":    
    model = ATTEEGNet()
    input_tensor = torch.randn(100, 1, 20, 500)
    output = model(input_tensor)
    print(output.shape)

    # model.visualize_attention(0)