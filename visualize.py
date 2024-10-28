import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from modules.EEGNET import EEGNet, ATTEEGNet
from random import randint
# model = ATTEEGNet()

model_path = '91_acc.pth'  # Replace with the actual path
model = torch.load(model_path).cuda()

data = np.load(r'clipped_data\mdd_control.npy')
x = data[randint(0, len(data))]
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
# Assuming model is an instance of ATTEEGNet and x is your input tensor
output = model(x)

# Retrieve the attention weights
attn_weights = model.attn_weights.detach().cpu().numpy()  # Shape: (batch_size, num_heads, height, height)

print(attn_weights)
print(attn_weights.shape)
# # Average across heads for a global view, or select a specific head
# avg_attn_weights = attn_weights.mean(axis=1)  # Shape: (batch_size, height, height)
# print(avg_attn_weights.shape)
# Visualize as a heatmap for one sample
# sns.heatmap(avg_attn_weights[0], cmap="viridis")
# plt.title("Attention Map")
# plt.xlabel("Sequence Position")
# plt.ylabel("Sequence Position")
# plt.show()
