from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import seaborn as sns
from modules.EEGNET import EEGNet
from modules.Att_EEGNET import Transformer_Model
from torch.utils.data import TensorDataset, DataLoader
import mne
from random import randint


names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Cz', 'Pz', 'A2']

montage = mne.channels.make_standard_montage('standard_1020')
all_positions = montage.get_positions()['ch_pos']
selected_positions = np.array([all_positions[name][:2] for name in names if name in all_positions])

pos = []

model_path = r'saved_models\fixed_transformer_best.pth'
state_dict = torch.load(model_path)
model = Transformer_Model(save_weights=True)
model.load_state_dict(state_dict)
model = model.cuda()

data = np.load(r'Leave_one_subject_out\Validation\mdd_patient.npy')
x_og = data[randint(0, len(data))]
x = torch.tensor(x_og, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
output = model(x)

weights = model.saved_weights.detach().cpu().numpy()

weights = weights.squeeze()
weights = weights.ravel()

print("Reshaped array shape:", weights.shape)

print(weights)
mne.viz.plot_topomap(weights, selected_positions, show=True)
