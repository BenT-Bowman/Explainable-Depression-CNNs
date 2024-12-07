import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch.utils.data import TensorDataset, DataLoader

def label_and_load_data():
    # TODO
    ...
def load_val_split(pairings: list, excluded_pairing: list):
    #
    # Training
    #
    mdd = []
    h   = []
    mdd_labels = []
    h_labels = []
    for pair in pairings:
        h_sub = next((s for s in pair if "H" in s), None)
        h.append(np.load(h_sub))
        h_labels.append( np.zeros((h[-1].shape[0])))
        mdd_sub = next((s for s in pair if "MDD" in s), None)
        mdd.append(np.load(mdd_sub))
        mdd_labels.append( np.ones((mdd[-1].shape[0])))
    

    h = np.concatenate(h, axis=0)
    mdd = np.concatenate(mdd, axis=0)
    h_labels = np.concatenate(h_labels, axis=0)
    mdd_labels = np.concatenate(mdd_labels, axis=0)

    training = np.concatenate([h, mdd], axis=0)
    training_labels = np.concatenate([h_labels, mdd_labels], axis=0)

    #
    # Validation
    #

    h_sub = next((s for s in excluded_pairing if "H" in s), None)
    h = np.load(h_sub)
    h_labels = np.zeros((h.shape[0]))
    mdd_sub = next((s for s in excluded_pairing if "MDD" in s), None)
    mdd = np.load(mdd_sub)
    mdd_labels = np.ones((mdd.shape[0]))

    validation = np.concatenate([h, mdd], axis=0)
    labels_val = np.concatenate([h_labels, mdd_labels], axis=0)


    return (training, training_labels), (validation, labels_val)



def select_indices(lst, indices):
    return [lst[i] for i in indices]

def create_Dataset(dataset, labels):
    # dataset, labels = training
    dataset = torch.Tensor(dataset).unsqueeze(1)
    labels = torch.Tensor(labels)
    return TensorDataset(dataset, labels)


def LOSOCV(path, sub_paths = ["Control", "Patient"]):
    if len(sub_paths) != 2:
        raise ValueError("Sub_path length should be equal to 2")
    files = []
    for sub_path in sub_paths:
        sub_dir = os.path.join(path, sub_path)
        files.append( sorted([os.path.join(sub_dir, f) for f in os.listdir(sub_dir)]))
    min_length = min(len(files[0]), len(files[1]))
    files[0] = files[0][:min_length]
    files[1] = files[1][:min_length]
    pairings = list(zip(files[0], files[1]))

    # Assign group IDs (one ID per pairing)
    groups = np.arange(len(pairings))

    # Initialize LeaveOneGroupOut
    logo = LeaveOneGroupOut()


    # Perform LOSOCV based on unique pairings
    for train_idx, test_idx in logo.split(pairings, groups=groups):
        # print("HEELL")
        # print(pairings[test_idx[0]])
        training, validation = load_val_split(select_indices(pairings, train_idx), pairings[test_idx[0]])
        training_dataset = create_Dataset(*training)
        validation_dataset = create_Dataset(*validation)

        yield training_dataset, validation_dataset

if __name__ == "__main__":
    # print("W")
    for thing in LOSOCV(r'train_data\leave_one_subject_out_CV'):
        ...