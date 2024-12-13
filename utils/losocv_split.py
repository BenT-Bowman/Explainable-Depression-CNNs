import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_val_split(pairings: list, excluded_pairing: list):
    def sub_func(targets):
        mdd = []
        h   = []
        mdd_labels = []
        h_labels = []
        for pair in targets:
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
        data = np.concatenate([h, mdd], axis=0)
        labels = np.concatenate([h_labels, mdd_labels], axis=0)
        # print(len(h), len(mdd))
        return data,  labels
    
    training = sub_func(pairings)
    validation = sub_func(excluded_pairing)


    return training, validation



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
        training, validation = load_val_split(select_indices(pairings, train_idx), [pairings[test_idx[0]]])
        training_dataset = create_Dataset(*training)
        validation_dataset = create_Dataset(*validation)

        yield training_dataset, validation_dataset

def LOSOSplit(path, sub_paths=["Control", "Patient"], validation_size=5, num_folds=18):
    if len(sub_paths) != 2:
        raise ValueError("Sub_path length should be equal to 2")
    if validation_size < 1:
        raise ValueError("Validation size must be at least 1")
    if num_folds < 1:
        raise ValueError("Number of folds must be at least 1")
    
    files = []
    for sub_path in sub_paths:
        sub_dir = os.path.join(path, sub_path)
        files.append(sorted([os.path.join(sub_dir, f) for f in os.listdir(sub_dir)]))
    
    min_length = min(len(files[0]), len(files[1]))
    files[0] = files[0][:min_length]
    files[1] = files[1][:min_length]
    pairings = list(zip(files[0], files[1]))

    total_pairs = len(pairings)
    
    # Check if validation size is larger than the number of pairs
    if validation_size > total_pairs:
        raise ValueError("Validation size cannot be greater than the total number of pairs")

    # Number of samples per fold, ensuring groups can overlap (double dip)
    for fold in range(num_folds):
        # Select validation set (can double dip)
        validation_start_idx = (fold * validation_size) % total_pairs
        validation_end_idx = (validation_start_idx + validation_size) % total_pairs
        if validation_end_idx > validation_start_idx:
            validation = pairings[validation_start_idx:validation_end_idx]
        else:
            # Wrapping around the list to allow for double dipping
            validation = pairings[validation_start_idx:] + pairings[:validation_end_idx]
        
        # Remaining pairs for training (excluding the validation set)
        training = [pair for pair in pairings if pair not in validation]

        # print("\n\n\n", validation)

        # Load training and validation data
        training, validation = load_val_split(training, validation)
        training_dataset = create_Dataset(*training)
        validation_dataset = create_Dataset(*validation)

        yield training_dataset, validation_dataset


if __name__ == "__main__":
    # print("W")
    i=0
    for thing in LOSOSplit(r'train_data\leave_one_subject_out_CV'):
        print(i)
        i+=1