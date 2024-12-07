import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from tqdm import tqdm
import argparse
import numpy as np 
from modules.EEGNET import EEGNet #, ATTEEGNet
from modules.Att_EEGNET import ATTEEGNet, Transformer_Model
from utils.training_utils import EarlyStop
from modules.CAEW import CAEW_EEGNet
from sklearn.metrics import accuracy_score

import os
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--model_type', '-t', type=str, required=False, default="EEGNet", help='Options: EEGNet, att_wavelet, att_EEGNet.')
    parser.add_argument('--num_epochs', '-e', type=int, required=False, default=100, help='Number of epochs to train models for.')
    parser.add_argument('--data_path', '-p',  type=str, required=True, help='Path to data.')
    parser.add_argument('--lr', type=float,       required=False,    default=1e-5)
    # parser.add_argument('--model_save_loc', -'m', type=float,       required=False,    default=0.0001)
    args = parser.parse_args()

    return args.model_type, args.num_epochs, args.data_path, args.lr
model_type, num_epochs, data_path, lr = argparse_helper()


# Yes I am lazy, how did you know?
save_dir = "saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
from random import randint
rand_name = randint(0, 2000)

early_stop = EarlyStop(save_dir, model_type.upper()+"_"+str(rand_name))

if str(model_type).upper() == "EEGNET":
    model = EEGNet().to(device)
elif str(model_type).upper() == "ATT_WAVELET":
    pass
elif str(model_type).upper() == "ATT_EEGNET":
    model = ATTEEGNet().to(device)
elif str(model_type).upper() == "TRANS_EEGNET":
    model = Transformer_Model().to(device)
elif str(model_type).upper() == "CAEW":
    model = CAEW_EEGNet().to(device)
elif str(model_type).upper() == "TEST":
    import torch.nn.functional as F
    class TEST(nn.Module):
        def __init__(self,):
            super().__init__()
            self.tester = nn.Linear(10_000, 1)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return F.sigmoid(self.tester(x))
    model = TEST().to(device)

else:
    raise ValueError("Invalid model type")

def organize_and_label(path: str, label: int):
    data = np.load(path)
    print(data.shape)
    tensor = torch.from_numpy(data).float()
    # tensor = torch.from_numpy(data[:, :19, :]).float()
    labels = torch.from_numpy(np.full(data.shape[0], label))
    return tensor, labels

datasets = []

for group in [(r'train_data\Leave_one_subject_out\Main\mdd_control.npy',       r'train_data\Leave_one_subject_out\Main\mdd_patient.npy'),
              (r'train_data\Leave_one_subject_out\Validation\mdd_control.npy', r'train_data\Leave_one_subject_out\Validation\mdd_patient.npy')]:
    dataset = []
    labels = []
    for idx, path in enumerate(group):
        data, label = organize_and_label(path, idx)
        dataset.append(data)
        labels.append(label)
    labels = torch.cat((labels[0], labels[1]), dim=0)
    labels = labels.view(labels.size(0)).type(torch.float32)
    dataset = torch.cat(dataset, dim=0)
    dataset = dataset.view(dataset.size(0), 1, dataset.size(1), dataset.size(2))

    dataset = TensorDataset(dataset, labels)
    datasets.append(dataset)




batch_size = 128
train_loader = DataLoader(dataset=datasets[0], batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=datasets[1], batch_size=batch_size, shuffle=True)

optimizer  = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

history_train = []
history_val = []

attn_weights_list = []
def save_attn_weights(module, input, output):
    try:
        attn_weights_list.append(module.transformer.layers[0].attn_weights)
        attn_weights_list.append(module.transformer.layers[1].attn_weights)
    except Exception as e:
        ...

model.register_forward_hook(save_attn_weights)

# import time
# import matplotlib.pyplot as plt

for epoch in range(num_epochs):
        
    model.train()
    pbar = tqdm(train_loader)
    

    running_loss = 0.0
    correct = 0
    total_batches = 0
    for batch_idx, (images, labels) in enumerate(pbar):

        attn_weights_list = []

        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).view(-1)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1
        pbar.set_description(f"Epoch {epoch+1}, Running Loss: {running_loss / (batch_idx + 1):.4f}")
        # plt.gcf().canvas.flush_events()
    # scheduler.step(running_loss/(batch_idx+1))
    
    history_train.append(running_loss/(batch_idx+1))
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for images, labels in tqdm(validation_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            

            preds = model(images).view(-1)
            loss = criterion(preds, labels)
            
            val_loss += loss.item()
            predicted = (preds >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            attn_weights_list = []
    # Calculate average validation loss and accuracy
    val_loss /= len(validation_loader)
    val_acc = correct / total
    history_val.append(val_loss)
    if early_stop(model, val_loss, 5):
        break

    del preds, loss, predicted
    torch.cuda.empty_cache()

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


plt.plot(history_val, label="Validation Loss")
plt.plot(history_train, label="Train Loss")
plt.show()