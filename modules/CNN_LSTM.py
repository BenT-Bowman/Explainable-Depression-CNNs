import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_lstm(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv1d(20, 5, kernel_size=64, stride=1)
        self.conv2 = nn.Conv1d(5, 3, kernel_size=128)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.2)
        # self.lstm1 = nn.LSTM( ,batch_first=True)
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        return x


if __name__ == "__main__":    

    model = cnn_lstm()
    input_tensor = torch.randn(100, 1, 20, 256)
    output = model(input_tensor)
    print(output.shape)