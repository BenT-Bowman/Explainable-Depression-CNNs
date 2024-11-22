import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CWT_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x):
        
        return x
    
if __name__ == "__main__":
    t = torch.randn(32, 1, 99, 500)
    print(t.shape)
    model = CWT_CNN()
    print(model(t).shape)