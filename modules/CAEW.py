import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
try:
    from .EEGNET import EEGNet
    from .DeprNet import DeprNet
    from .NeuromodulatedAttn import TransformerEncoderLayerWithAttn
except Exception as e:
    from NeuromodulatedAttn import TransformerEncoderLayerWithAttn
    from EEGNET import EEGNet
    from DeprNet import DeprNet


class CAEW(nn.Module):
    def __init__(self, num_heads=10, is_neuromod=False):
        super().__init__()

        # Convolution Block 1
        self.t_conv1 = nn.Conv2d(1, 16, (1, 16), padding=(0, 8), bias=False)
        self.t_batchnorm1 = nn.BatchNorm2d(16)
        self.activation1 = nn.ELU()

        # Separable + Dilated Convolution Block 2
        self.depthwise2 = nn.Conv2d(16, 16, (1, 8), padding=(0, 7), dilation=(1, 8), groups=16, bias=False)  # Depthwise
        self.pointwise2 = nn.Conv2d(16, 32, (1, 1), bias=False)  # Pointwise
        self.t_batchnorm2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.25)
        self.activation2 = nn.ELU()

        # Convolution Block 3 (Reduction)
        self.t_conv3 = nn.Conv2d(32, 1, (1, 10), bias=False) # TODO: Consider separable stuff here too
        self.t_batchnorm3 = nn.BatchNorm2d(1)
        self.dropout3 = nn.Dropout(0.25)
        self.activation3 = nn.ELU()

        # Transformer Encoder
        if is_neuromod:
            self.transformer = nn.TransformerEncoder(
                TransformerEncoderLayerWithAttn(320, num_heads, batch_first=True), 2
            )
        else:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(320, num_heads, batch_first=True), 2
            )

    def forward(self, x):
        # Convolution Block 1
        x = self.t_conv1(x)
        x = self.t_batchnorm1(x)
        x = self.activation1(x)
        x = F.avg_pool2d(x, (1, 4))

        # Separable + Dilated Convolution Block 2
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.t_batchnorm2(x)
        x = self.dropout2(x)
        x = self.activation2(x)
        x = F.avg_pool2d(x, (1, 8))

        batch_size, filters, channels, seq_length = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch_size, channels, seq_length * filters)
        x = self.transformer(x)
        x = x.view(batch_size, channels, filters, seq_length).permute(0, 2, 1, 3)

        # print(x.shape)
        
        x = self.t_conv3(x)
        x = self.t_batchnorm3(x)
        x = self.dropout3(x)
        x = F.softmax(x, 2)
        return x

class CAEW_EEGNet(EEGNet):
    def __init__(self, num_channels=20, num_heads=8):
        super().__init__(num_channels)

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
        
class CAEW_Alone(nn.Module):
    def __init__(self ):
        super().__init__()
        self.caew = CAEW()
        self.caew_weights = None

        self.linear = nn.Linear(10_000, 1)
    
    def forward(self, x):
        self.caew_weights = self.caew(x)
        x = x*self.caew_weights
        x = x.view(x.size(0), -1)
        return F.sigmoid(self.linear(x))

class CAEW_DeprNet(DeprNet):
    def __init__(self, num_heads=8):
        super().__init__()
        self.caew = CAEW(num_heads=num_heads)
        self.caew_weights = None
    def forward(self, x):
        self.caew_weights = self.caew(x)
        x = x*self.caew_weights

        x = self.model(x)
        return F.sigmoid(x)
        
if __name__ == "__main__":    
    from time import time

    # model = EEGNet(num_channels=20)
    # input_tensor = torch.randn(100, 1, 20, 500)
    # t_0 = time()
    # output = model(input_tensor)
    # t_0 = time()-t_0
    # print(output.shape, t_0)

    # model = CAEW_EEGNet(num_channels=20)
    # input_tensor = torch.randn(100, 1, 20, 500)
    # t_0 = time()
    # output = model(input_tensor)
    # t_0 = time()-t_0
    # print(output.shape, t_0)

    model = CAEW_EEGNet()    
    input_tensor = torch.randn(100, 1, 20, 500)
    t_0 = time()
    output = model(input_tensor)
    t_0 = time()-t_0
    print(output.shape, t_0)

    # model = CAEW_DeprNet()    
    # input_tensor = torch.randn(100, 1, 20, 500)
    # t_0 = time()
    # output = model(input_tensor)
    # t_0 = time()-t_0
    # print(output.shape, t_0)