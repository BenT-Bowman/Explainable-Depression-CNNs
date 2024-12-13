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
from modules.CAEW import CAEW_EEGNet
from utils.losocv_split import LOSOCV, LOSOSplit
from torch.utils.data import TensorDataset, DataLoader


if __name__ == "__main__":
    data_path = r"train_data\leave_one_subject_out_CV"
    for fold_idx, (training, validation) in enumerate(LOSOSplit(path=data_path)):
        val_loader = DataLoader(dataset=validation, batch_size=128, shuffle=True)
    #
    # Paths
    #
    model_path = r'saved_models\final_models\CAEW_584\fold_10_CAEW_584'

    #
    # Import Model
    #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(model_path)
    model = CAEW_EEGNet()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Target Layer
    target_layer = model.separableConv[0]

    #
    # Import numpy
    #

            # Target Layer
    target_layer = model.separableConv[0]

    # Import numpy
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        print(labels[0])
        x_og = inputs[0][0].squeeze(0).numpy()
        print(x_og.shape)
        selected_positions = cr.find_selected()

        # reeg.show_eeg(x_og)
        cr.channel_rep_show(x_og, model, selected_positions)
        # gc.grad_cam_rep(x_og, model, target_layer)
        break

    # exit()

    # data = np.load(np_path)
    # x_og = data[randint(0, len(data))]

    # selected_positions = cr.find_selected()

    # reeg.show_eeg(x_og)
    # cr.channel_rep_show(x_og, model, selected_positions)
    # gc.grad_cam_rep(x_og, model, target_layer)