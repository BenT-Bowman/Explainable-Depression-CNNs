import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from utils.training_utils import EarlyStop
from modules.EEGNET import EEGNet
from modules.Att_EEGNET import ATTEEGNet, Transformer_Model
from modules.CAEW import CAEW_EEGNet, CAEW_DeprNet, CAEW_Alone
from utils.losocv_split import LOSOCV, LOSOSplit
from modules.DeprNet import DeprNet, NeuromodulatedDeprNet

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import numpy as np 
import os
import matplotlib.pyplot as plt
from random import randint

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
class TEST(nn.Module):
    def __init__(self,):
        super().__init__()
        self.tester = nn.Linear(10_000, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.sigmoid(self.tester(x))

# data_path = str(input("LOGO-CV data path: "))
data_path = r"train_data\leave_one_subject_out_CV"
model_path = r"saved_models\final_models\AAEW__496"



data = os.listdir(model_path)

def function(x):
    return int(x.split('_')[1])

sorted_data = sorted(data, key=lambda x: function(x))

print(sorted_data)
device = "cuda" if torch.cuda.is_available() else "cpu"

def metric_calc(labels, preds):
    """
    Calculate and print accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return accuracy, precision, recall, f1
    
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
metrics = []
# Iterate through the folds
for fold_idx, (training, validation) in enumerate(LOSOSplit(path=data_path)):
    m_path = sorted_data[fold_idx]
    m_path = os.path.join(model_path, m_path)

    # Load model state
    state_dict = torch.load(m_path)
    # model = CAEW_EEGNet()
    model = CAEW_Alone()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Collect true labels and predictions for validation
    all_labels = []
    all_preds = []
    model.eval()  # Ensure the model is in evaluation mode

    val_loader = DataLoader(dataset=validation, batch_size=128, shuffle=True)
    
    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in val_loader:  # Assuming validation is a DataLoader
            # print(labels.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            # preds = torch.argmax(outputs, dim=1)
            preds = (outputs >= 0.5).float()
            # print(preds)
            
            # Store predictions and true labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics using sklearn
    metrics.append(metric_calc(all_labels, all_preds))

with open("text.txt", "w") as f:
    f.write(f"Accuracy: {[metric[0] for metric in metrics]}\n")
    f.write(f"Precision: {[metric[1] for metric in metrics]}\n")
    f.write(f"Recall: {[metric[2] for metric in metrics]}\n")
    f.write(f"F1: {[metric[3] for metric in metrics]}\n")