import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # Self-attention with output attn_weights
        src2, self.attn_weights = self.self_attn(src, src, src, 
                                            attn_mask=src_mask, 
                                            key_padding_mask=src_key_padding_mask,
                                            need_weights=True)
        # Apply dropout and add & norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # Return output and attention weights
        return src#, self.attn_weights
    
    
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
        # self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(TransformerEncoderLayerWithAttn(32, num_heads, batch_first=True), 2)

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
        # Feature Extraction
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Transformer
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, height * channels)
        x = self.transformer(x)
        x = x.view(batch_size, width, channels, height).permute(0, 2, 3, 1)

        # Dimension Reduction
        x = F.avg_pool2d(x, (1, 4))
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