import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim

from tqdm import tqdm
import argparse
import numpy as np

from modules.EEGNET import EEGNet #, ATTEEGNet
from modules.Att_EEGNET import ATTEEGNet
from sklearn.metrics import accuracy_score

import os
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--model_type', type=str, required=False, default="EEGNet", help='Options: EEGNet, att_wavelet, att_EEGNet.')
    parser.add_argument('--num_epochs', type=int, required=False, default=100, help='Number of epochs to train models for.')
    parser.add_argument('--data_path',  type=str, required=True, help='Path to data.')
    parser.add_argument('--lr', type=float,       required=False,    default=0.0001)
    args = parser.parse_args()

    return args.model_type, args.num_epochs, args.data_path, args.lr

# Yes I am lazy, how did you know?
save_dir = "saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
last_loss = np.inf
since_last = 0

def early_stop(model, val_loss, patience, model_name="best_model.pth"):
    global last_loss
    global since_last
    model_path = os.path.join(save_dir, model_name)
    
    if val_loss < last_loss:
        last_loss = val_loss
        torch.save(model.state_dict(), model_path)
        since_last = 0
    else:
        print(f"Hasn't improved in {since_last}")
        since_last += 1
    
    if since_last > patience:
        return True
    return False


# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots() 


def show(model):
    ax.clear()  # Clear the current axis
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.norm().item())

    # Create a plot of the gradient norms
    ax.plot(gradients)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms During Training')
    fig.canvas.draw()  # Draw the plot
    fig.canvas.flush_events()  # Flush the events
    plt.pause(0.001)  # Pause for a short time to allow the plot to update


model_type, num_epochs, data_path, lr = argparse_helper()

if str(model_type).upper() == "EEGNET":
    model = EEGNet().to(device)
    # model = TEST().to(device)
elif str(model_type).upper() == "ATT_WAVELET":
    pass
elif str(model_type).upper() == "ATT_EEGNET":
    model = ATTEEGNet().to(device)

else:
    raise ValueError("Invalid model type")


# TODO: Fix Dataloaders

# patient = np.load(r'Leave_one_subject_out\Main\mdd_patient.py')

def organize_and_label(path: str, label: int):
    data = np.load(path)
    tensor = torch.from_numpy(data).float()
    labels = torch.from_numpy(np.full(data.shape[0], label))
    return tensor, labels

# control = organize_and_label(r'Leave_one_subject_out\Main\mdd_control.py')
# patient = organize_and_label(r'Leave_one_subject_out\Main\mdd_patient.py')
# val_control = organize_and_label(r'Leave_one_subject_out\Main\mdd_control.py')
# val_patient = organize_and_label(r'Leave_one_subject_out\Main\mdd_patient.py')

datasets = []

for group in [(r'Leave_one_subject_out\Main\mdd_control.npy', r'Leave_one_subject_out\Main\mdd_patient.npy'),
              (r'Leave_one_subject_out\Validation\mdd_control.npy', r'Leave_one_subject_out\Validation\mdd_patient.npy')]:
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
# print(type(datasets[0]),"\n"*2, datasets[0])
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


for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader)
    
    results = []
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

    model.eval()  # Set the model to evaluation mode
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
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
    # Calculate average validation loss and accuracy
    val_loss /= len(validation_loader)
    val_acc = correct / total
    history_val.append(val_loss)
    if early_stop(model, val_loss, 15):
        break

    del preds, loss, predicted
    torch.cuda.empty_cache()

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


plt.plot(history_val)
plt.show()


# torch.save(saving_model, f"{input('Model Name: ')}.pth")