import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim

from tqdm import tqdm
import argparse
import numpy as np

from modules.EEGNET import EEGNet, ATTEEGNet
from sklearn.metrics import accuracy_score

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
saving_model = None
last_loss = np.inf
since_last = 0

def early_stop(model, val_loss, patience):
    global saving_model
    global last_loss
    global since_last
    if val_loss < last_loss:
        last_loss = val_loss
        saving_model = model
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

control = np.load(r'new_data\mdd_control.npy')
patient = np.load(r'new_data\mdd_patient.npy')

total_samples = patient.shape[0]

# Select 9130 random indices
random_indices = np.random.choice(total_samples, size=9130, replace=False)

# Select the corresponding data
patient = patient[random_indices, :, :]
print(control.shape, patient.shape)

control = torch.from_numpy(control).float()
patient = torch.from_numpy(patient).float()

control_labels = torch.from_numpy(np.full(control.shape[0], 0))
patient_labels = torch.from_numpy(np.full(patient.shape[0], 1))

dataset = torch.cat((control, patient), dim=0)
dataset = dataset.view(dataset.size(0), 1, dataset.size(1), dataset.size(2))
labels = torch.cat((control_labels, patient_labels), dim=0)
labels = labels.view(labels.size(0)).type(torch.float32)
dataset = TensorDataset(dataset, labels)

# Being Economical :) 
del labels
del control
del control_labels
del patient
del patient_labels

validation_size = int(len(dataset) * 0.2)
train_size = len(dataset) - validation_size

train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


optimizer  = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

history_train = []
history_val = []




for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader)
    
    results = []
    running_loss = 0.0
    correct = 0
    total_batches = 0
    for batch_idx, (images, labels) in enumerate(pbar):
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

       
        # # Check gradients
        # for param in model.parameters():
            # if param.grad is not None:
            #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #         raise ValueError("NaN or infinity detected in the gradients")
            # else:
            #     print("NONE")
        # show(model)

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            

            preds = model(images).view(-1)
            loss = criterion(preds, labels)
            
            val_loss += loss.item()

            predicted = (preds >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        

        # for i in range(5):
        #     model.visualize_attention(0)
            
    # Calculate average validation loss and accuracy
    val_loss /= len(validation_loader)
    val_acc = correct / total
    history_val.append(val_loss)
    if early_stop(model, val_loss, 5):
        break
    
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


plt.plot(history_val)
plt.show()


torch.save(saving_model, f"{input('Model Name: ')}.pth")