import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.Att_EEGNET import Transformer_Model

import numpy as np
from random import randint

import torch.nn.functional as F
import cv2

#
# Paths
#

np_path=r'Leave_one_subject_out\Validation\mdd_control.npy'
model_path = r'saved_models\25_a.pth'

#
# Import Model
#

device = "cuda" if torch.cuda.is_available() else "cpu"

state_dict = torch.load(model_path)
model = Transformer_Model(save_weights=True)
model.load_state_dict(state_dict)
model = model.to(device)

# Target Layer
target_layer = model.separableConv[0]

#
# Import numpy
#

data = np.load(np_path)
x_og = data[randint(0, len(data))]
x = torch.tensor(x_og, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

#
# Forward hook to grab feature maps
#

feature_maps = []

def forward_hook(module, input, output):
    feature_maps.append(output)

hook = target_layer.register_forward_hook(forward_hook)

model.eval()
output = model(x)

target_class_score = output[0]



#
# Backward Pass to Compute Gradients
#

model.zero_grad()
target_class_score.backward()
gradients = target_layer.weight.grad
feature_maps = feature_maps[0].squeeze(0)

print(f"Gradients shape: {gradients.shape}") 
print(f"Feature maps shape: {feature_maps.shape}")
print(f"Largest feature number {feature_maps.max()}\nSmallest feature number {feature_maps.min()}")

weights = gradients.mean(dim=(-1), keepdim=True) 
weights = weights.squeeze(1)
print(f"Weights shape: {weights.shape}")
cam = (weights * feature_maps).sum(dim=0)
relu_cam = F.relu(cam)
leaky_cam = F.leaky_relu(cam)




def testing_cam(cam):
    print(f"\033[31mCam Largest: {cam.max()}\nCam Smallest: {cam.min()}\033[37m")
    cam = cam - cam.min()
    cam = cam / cam.max()

    print(f"\033[34mCam Largest: {cam.max()}\nCam Smallest: {cam.min()}\033[37m")

    print(f"CAM shape: {cam.shape}")  
    heatmap = cam.cpu().data.numpy()
    print(f"Heatmap shape: {heatmap.shape}")  
    print(f"Original Shape: {x.shape}")
    heatmap = cv2.resize(heatmap, (x.shape[3], x.shape[2]))

    print(f"Resized heatmap shape: {heatmap.shape}")

    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(heatmap, cmap='jet', aspect='auto', interpolation='bilinear')
    plt.colorbar()
    plt.title("Grad-CAM Heatmap")
    plt.xlabel("Time Step")
    plt.ylabel("Importance")
    plt.show()
testing_cam(cam)