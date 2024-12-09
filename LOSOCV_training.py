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

device = "cuda" if torch.cuda.is_available() else "cpu"

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_path', '-p',  type=str, required=True, help='Path to data.')
    parser.add_argument('--model_type', '-t', type=str, required=False, default="EEGNet", help='Options: EEGNet, att_wavelet, att_EEGNet.')
    parser.add_argument('--model_save_loc', '-m', type=str, required=False, default="saved_models")
    parser.add_argument('--num_epochs', '-e', type=int, required=False, default=100, help='Number of epochs to train models for.')
    parser.add_argument('--batch_size', '-b', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--cross_entropy', required=False, default=False, action='store_true')
    
    args = parser.parse_args()

    return args.model_type, args.num_epochs, args.data_path, args.lr, args.model_save_loc, args.batch_size
model_type, num_epochs, data_path, lr, save_dir, batch_size = argparse_helper()

# print(type(cross_entropy))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if str(model_type).upper() == "EEGNET":
    model_class = EEGNet #().to(device)
elif str(model_type).upper() == "ATT_WAVELET":
    pass
elif str(model_type).upper() == "ATT_EEGNET":
    model_class = ATTEEGNet #().to(device)
elif str(model_type).upper() == "TRANS_EEGNET":
    model_class = Transformer_Model #().to(device)
elif str(model_type).upper() == "CAEW":
    model_class = CAEW_EEGNet #().to(device)
elif str(model_type).upper() == "DEPR":
    model_class = DeprNet
elif str(model_type).upper() == "CEPR":
    model_class = CAEW_DeprNet
elif str(model_type).upper() == "NEPR":
    model_class = NeuromodulatedDeprNet
elif str(model_type).upper() == "AEPR":
    model_class = CAEW_Alone
elif str(model_type).upper() == "TEST":
    import torch.nn.functional as F
    class TEST(nn.Module):
        def __init__(self,):
            super().__init__()
            self.tester = nn.Linear(10_000, 1)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return F.sigmoid(self.tester(x))
    model_class = TEST #().to(device)
else:
    raise ValueError("Invalid model type")

print(type(lr))
rand_name_id = randint(0, 2000)
accuracies = []

for fold_idx, (training, validation) in enumerate(LOSOSplit(path=data_path)):
    model = model_class().to(device)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 
    early_stop = EarlyStop(save_dir, "fold_" + str(fold_idx) + "_" + model_type.upper()+"_"+str(rand_name_id))

    train_loader = DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)

    history_train = []
    history_val = []
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader) # We love tqdm :)

        running_loss = 0.0
        correct = 0
        total_batches = 0
        for batch_idx, (data, labels) in enumerate(pbar):
            attn_weights_list = []

            data = data.to(device)
            labels = labels.to(device)

            preds = model(data).view(-1)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1
            pbar.set_description(f"Epoch {epoch+1}, Running Loss: {running_loss / (batch_idx + 1):.4f}")
        history_train.append(running_loss/(batch_idx+1))
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): 
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                

                preds = model(images).view(-1)
                loss = criterion(preds, labels)
                
                val_loss += loss.item()
                predicted = (preds >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total
        history_val.append(val_loss)
        if early_stop(model, val_loss, val_acc, 5):
            break
        del preds, loss, predicted
        torch.cuda.empty_cache()
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    accuracies.append(early_stop.last_acc)
    print(f"{accuracies=}")