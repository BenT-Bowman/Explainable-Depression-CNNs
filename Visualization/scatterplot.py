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
from modules.Att_EEGNET import ATTEEGNet, Transformer_Model
from torch.utils.data import TensorDataset, DataLoader

#
# Load state_dict and create model
#

def load_model(model_path: str, model_class, device = "cuda:0"):
    """
    Load state dict, create model with model_class
    """
    state_dict = torch.load(model_path)
    model = model_class()
    model.load_state_dict(state_dict)
    return model.to(device)

EEGNet_transformer = load_model(r'saved_models\fixed_transformer_best.pth', Transformer_Model)
EEGNet_base = load_model(r'saved_models\EEGNet_base.pth', EEGNet)

#
# Load Validation Data
#

def organize_and_label(path: str, label: int):
    data = np.load(path)
    tensor = torch.from_numpy(data).float()
    labels = torch.from_numpy(np.full(data.shape[0], label))
    return tensor, labels

def dataset_loader(targets: list):
    datasets = []
    for group in [targets]:
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
    return datasets


validation_path = (r'Leave_one_subject_out\Validation\mdd_control.npy', r'Leave_one_subject_out\Validation\mdd_patient.npy')
dataset = dataset_loader(validation_path)
validation_loader = DataLoader(dataset=dataset[0], batch_size=128, shuffle=True)

#
# Validation loop
#

def validation_loop(model, validation_loader, device="cuda:0"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_features = []

    all_preds = []
        
    with torch.no_grad(): 
        for images, labels in tqdm(validation_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            preds, features = model(images, save_features=True)
            preds = preds.view(-1)
            # loss = criterion(preds, labels)s
            
            # val_loss += loss.item()
            predicted = (preds >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu().numpy())
            all_features.append(features.detach().cpu())

    # Calculate average validation loss and accuracy
    val_loss /= len(validation_loader)
    val_acc = correct / total
    print(type(model), val_acc)
    return np.concatenate(all_labels), torch.cat(all_features, dim=0).numpy(), torch.cat(all_preds, dim=0).numpy()
    
#
# Feature Capturing
#

base_labels, base_features, base_preds = validation_loop(EEGNet_base, validation_loader)
tran_labels, trans_features, tran_preds = validation_loop(EEGNet_transformer, validation_loader)

print("base_labels", base_labels.shape)
print("tran_labels", tran_labels.shape)


print("base_features", base_features.shape)
print("tran_features", trans_features.shape)

print("base_preds", base_preds.shape)
print("tran_preds", tran_preds.shape)

#
# Dimensionality Reduction
#

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def reduction(data):
    # pca = PCA(n_components=100)
    tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=1000)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # data_scaled = pca.fit_transform(data_scaled)

    return tsne.fit_transform(data_scaled)




base_tsne = reduction(base_features)
tran_tsne = reduction(trans_features)

#
# Display
#

def display_scatterplot(data, labels):
    colors = np.where(labels == 1, 'blue', 'red')


    legend_labels = {
        'red': 'Control (0)',
        'blue': 'Patient (1)',
    }


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    for color, label in legend_labels.items():
        ax.scatter([], [], [], c=color, label=label)

    ax.legend(loc="upper right")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, s=10, alpha=0.7)
    ax.set_title("3D t-SNE Visualization")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    plt.show()

display_scatterplot(base_tsne, base_labels)
display_scatterplot(tran_tsne, tran_labels)

display_scatterplot(base_tsne, base_preds)
display_scatterplot(tran_tsne, tran_preds)