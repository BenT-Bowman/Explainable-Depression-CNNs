import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

# Directory paths
control_dir = "train_data\leave_one_subject_out_CV\Control"
patient_dir = "train_data\leave_one_subject_out_CV\Patient"

# Load file names
control_files = sorted([os.path.join(control_dir, f) for f in os.listdir(control_dir)])
patient_files = sorted([os.path.join(patient_dir, f) for f in os.listdir(patient_dir)])

# Ensure equal number of files
min_length = min(len(control_files), len(patient_files))
control_files = control_files[:min_length]
patient_files = patient_files[:min_length]

# Create unique pairings (one-to-one)
pairings = list(zip(control_files, patient_files))

# Assign group IDs (one ID per pairing)
groups = np.arange(len(pairings))

# Example dataset: Simulate some data corresponding to pairings
data = np.random.randn(len(pairings), 10)  # Example feature data
labels = np.random.randint(0, 2, len(pairings))  # Example binary labels

# Initialize LeaveOneGroupOut
logo = LeaveOneGroupOut()

# Perform LOSOCV based on unique pairings
for train_idx, test_idx in logo.split(data, labels, groups=groups):
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    # Example: Print the pairing left out
    print(f"Left out pairing: {pairings[test_idx[0]]}")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print("-" * 40)
