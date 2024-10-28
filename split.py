import os
import numpy as np
from tqdm import tqdm
import argparse
import re
import mne

##
# Utility Functions
##
def sliding_window(full_signal: np.ndarray, window: int = 1000, features: int = 20, original_length: int = 76800, skip: int = 250):
    # assert full_signal.shape == (features, original_length)
    for idx in range(0, len(full_signal[-1]), skip):
        signal = full_signal[:, idx:idx+window]
        # print(signal.shape)
        if signal.shape != (features, window):
            break
        yield signal

def full(data: np.ndarray, **kwargs) -> np.ndarray:
    temp = []
    for signal in tqdm(data):
        for result_window in sliding_window(signal, **kwargs):
            temp.append(result_window)

    return np.asarray(temp)

def scale(data: np.ndarray)->np.ndarray:
    data_array = []
    compare = 0.04
    for signal in tqdm(data):
        if signal.max() > compare or abs(signal.min()) > compare:
            continue
        data_array.append(signal)
    data_array = np.asarray(data_array)
    print(data_array.mean(), data_array.std(), data_array.shape)
    return data_array

def scale_neg_one(data: np.ndarray)->np.ndarray:
    """Rescale data """
    data = data - data.min()
    return (data / data.max()) * 2 -1

##
# Main
##
def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the input file. Current data parsing capability limited.')
    parser.add_argument('--save_file_directory', type=str, required=True, help='Directory to save the output file')
    parser.add_argument('--skip', type=int, default=50, help='Number of lines to skip (default: 50)')
    parser.add_argument('--seq_length', type=int, default=500, help='Sequence length (default: 500)')
    args = parser.parse_args()

    return args.data_file_path, args.save_file_directory, args.skip, args.seq_length

def main():
    file_path, save_file_directory, skip, seq_length = argparse_helper()
    # List all files in the directory
    file_list = os.listdir(file_path)

    # Filter DAT files
    edf_files = [file for file in file_list if file.endswith('.edf')]

    # Display the list of DAT files
    if not edf_files:
        print("No EDF files found in the directory.")
    else:
        print("EDF files in the directory:")
        for file in edf_files:
            print(file)

    patient = []
    control = []
    for file in edf_files:
        if not ('E' in file and 'C' in file):
            continue

        if file.startswith('H'):
            # patient.append(np.loadtxt(os.path.join(file_path, file)))
            print(file)
            raw = mne.io.read_raw_edf(os.path.join(file_path, file))

            # Get the EEG data
            data = raw.get_data()
            control.append(data)
            print(data.shape)
        else:
            # control.append(np.loadtxt(os.path.join(file_path, file)))
            print(file)
            raw = mne.io.read_raw_edf(os.path.join(file_path, file))

            # Get the EEG data
            data = raw.get_data()
            patient.append(data)
            print(data.shape)
    print(len(control))
    control = full(control, skip=skip, window=seq_length)
    patient = full(patient, skip=skip, window=seq_length)
    # print(control[0])
    # patient = scale_neg_one(scale(patient))
    # np.save(fr'{save_file_directory}/patient',patient )
    np.save(fr'{save_file_directory}/mdd_control',control )
    np.save(fr'{save_file_directory}/mdd_patient',patient )
    print("Finished: ",control.shape)
    print("Finished: ",patient.shape)





if __name__ == "__main__":
    main()