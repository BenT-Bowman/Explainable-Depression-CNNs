import channel_representation as cr
import raw_eeg as reeg
import grad_cam as gc

from random import randint
import torch
import numpy as np
import mne

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.Att_EEGNET import Transformer_Model, legacy

if __name__ == "__main__":
    #
    # Paths
    #

    np_path=r'train_data\Leave_one_subject_out\Validation\mdd_control.npy'
    model_path = r'saved_models\12_3_legacy.pth'

    #
    # Import Model
    #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(model_path)
    model = legacy(save_weights=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Target Layer
    target_layer = model.separableConv[0]

    #
    # Import numpy
    #

    data = np.load(np_path)
    x_og = data[randint(0, len(data))]

    selected_positions = cr.find_selected()

    reeg.show_eeg(x_og)
    cr.channel_rep_show(x_og, model, selected_positions)
    gc.grad_cam_rep(x_og, model, target_layer)