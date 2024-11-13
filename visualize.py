import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from modules.EEGNET import EEGNet#, ATTEEGNet
from modules.Att_EEGNET import ATTEEGNet
from random import randint
import seaborn as sns
from matplotlib.widgets import Slider
def plot_attention_on_raw_series(raw_input_data, attn_weights):
    """
    Plot raw time-series data with an interactive overlay of attention weights.
    
    Parameters:
    - raw_input_data: 2D tensor/array of the raw time-series data (e.g., [num_channels, seq_len]).
    - attn_weights: 2D or 3D tensor/array of attention weights. If 3D, they will be averaged across heads.
    """
    
    # If attn_weights is 3D (num_heads, seq_len, seq_len), average over heads
    if len(attn_weights.shape) == 3:
        attn_weights = attn_weights.mean(dim=0)  # Shape becomes [seq_len, seq_len]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create an initial plot of raw data (all channels) before interaction
    for eeg_channel in raw_input_data:
        ax.plot(eeg_channel, color="black", alpha=0.7)
    
    # Normalize attention weights for the entire sequence
    def update_plot(val):
        # Get the current time step from the slider
        current_time_step = int(val)

        # Focus on the current time step's attention (row of attention matrix)
        attention_focus = attn_weights[current_time_step, :]
        
        # Normalize the attention weights to [0, 1]
        attention_focus = (attention_focus - attention_focus.min()) / (attention_focus.max() - attention_focus.min())
        
        # Clear the plot and redraw it
        ax.clear()

        # Redraw EEG signal (all channels)
        for eeg_channel in raw_input_data:
            ax.plot(eeg_channel, color="black", alpha=0.3)

        # Overlay the attention weights for the current time step
        ax.fill_between(range(len(attention_focus)), attention_focus, color='red', alpha=0.5, label='Attention Weight')

        # Set the labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('EEG Signal Value')
        ax.set_title(f'Attention Overlay on Raw Time-Series Data (Time Step {current_time_step})')
        
        # Redraw the plot
        fig.canvas.draw()

    # Add a slider widget to control the time step
    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Time Step', 0, len(raw_input_data[0]) - 1, valinit=0, valstep=1)
    slider.on_changed(update_plot)

    # Show the plot
    plt.show()

attn_weights_list = []
def save_attn_weights(module, input, output):
    attn_weights_list.append((module.transformer.layers[0].attn_weights, module.transformer.layers[1].attn_weights))
    # attn_weights_list.append()

model_path = r'EEGNet_transformer.pth'  # Replace with the actual path
model = torch.load(model_path).cuda()


model.register_forward_hook(save_attn_weights)


data = np.load(r'clipped_data\mdd_control.npy')
x_og = data[randint(0, len(data))]
x = torch.tensor(x_og, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
# Assuming model is an instance of ATTEEGNet and x is your input tensor
output = model(x)

# Retrieve the attention weights
attn_weights = attn_weights_list[0][1].detach().cpu().numpy()[0, :, :] # model.attn_weights.detach().cpu().numpy()[0, :, :]  # Shape: (batch_size, num_heads, height, height)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(torch.from_numpy(attn_weights).unique())
print("Attention Weights Shape:", attn_weights.shape)
attn_weights_np = attn_weights.cpu().numpy() if isinstance(attn_weights, torch.Tensor) else attn_weights
plot_attention_on_raw_series(x_og, attn_weights)



# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(attn_weights_np, cmap="viridis", cbar=True)
# plt.title("Attention Heatmap")
# plt.xlabel("Sequence Position")
# plt.ylabel("Sequence Position")
# plt.show()
# exit()
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