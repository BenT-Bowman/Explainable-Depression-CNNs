import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from modules.EEGNET import EEGNet, ATTEEGNet
from random import randint
# model = ATTEEGNet()

model_path = '10_28_24_fixed_attn.pth'  # Replace with the actual path
model = torch.load(model_path).cuda()

data = np.load(r'clipped_data\mdd_control.npy')
x = data[randint(0, len(data))]
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
# Assuming model is an instance of ATTEEGNet and x is your input tensor
output = model(x)

# Retrieve the attention weights
attn_weights = model.attn_weights.detach().cpu().numpy()[0, :, :]  # Shape: (batch_size, num_heads, height, height)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(torch.from_numpy(attn_weights).unique())
print("Attention Weights Shape:", attn_weights.shape)
attn_weights_np = attn_weights.cpu().numpy() if isinstance(attn_weights, torch.Tensor) else attn_weights

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attn_weights_np, cmap="viridis", cbar=True)
plt.title("Attention Heatmap")
plt.xlabel("Sequence Position")
plt.ylabel("Sequence Position")
plt.show()
# print(attn_weights)

# If `width` was meant to be >1 and it's coming out as 1, check MHA input dimensions
# if attn_weights.shape[-2] == 1 or attn_weights.shape[-1] == 1:
#     print("Sequence length in MHA is 1; cannot create a meaningful heatmap.")
# else:
#     avg_attn_weights = attn_weights.mean(axis=1)  # Average across heads
#     print("Averaged Attention Weights Shape:", avg_attn_weights.shape)
#     sns.heatmap(avg_attn_weights[0], cmap="viridis")
#     plt.title("Attention Map")
#     plt.xlabel("Sequence Position")
#     plt.ylabel("Sequence Position")
#     plt.show()

# import torch.nn.functional as F

# # Assuming original_length is the original input's length
# original_length = 500
# attn_weights_resized = F.interpolate(torch.tensor(attn_weights).unsqueeze(0).unsqueeze(0), size=(original_length, original_length), mode='bilinear').squeeze()
# # print(attn_weights_resized.shape)

# # Retrieve the attention weights
# # attn_weights_resized = model.attn_weights.detach().cpu().numpy()  # Shape: (batch_size, num_heads, height, height)
# # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print("Attention Weights Shape:", attn_weights_resized.shape)
# attn_weights_resized_np = attn_weights_resized.cpu().numpy() if isinstance(attn_weights_resized, torch.Tensor) else attn_weights_resized

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(attn_weights_resized_np, cmap="viridis", cbar=True)
# plt.title("Attention Heatmap")
# plt.xlabel("Sequence Position")
# plt.ylabel("Sequence Position")
# plt.show()