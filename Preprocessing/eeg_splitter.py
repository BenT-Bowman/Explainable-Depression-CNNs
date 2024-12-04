import os
import numpy as np
from tqdm import tqdm
import argparse
import mne

#
# Preprocessing Helper Functions
#

def read_eeg(file_path:str, file:str)->np.ndarray:
    """
    Reads and converts the data in `file_path` to a numpy array.
    """
    if not "EC" in file:
        raise ValueError("Only Include files with Eyes Closed")
    
    return mne.io.read_raw_edf(os.path.join(file_path, file)).get_data()

def slide_window(matrix, k, skip):
    """
    Slide a window of size k over a matrix of shape (20, n).

    Args:
        matrix (numpy array): Matrix of shape (20, n)
        k (int): Window size

    Returns:
        numpy array: Slice of shape (20, k)
    """
    n = matrix.shape[1]
    for i in range(0, (n - k + 1), skip):
        yield matrix[:20, i:i+k]

def normalize(arr: np.ndarray):
    min_vals = np.min(arr)
    max_vals = np.max(arr)

    return (arr - min_vals) / (max_vals - min_vals) * 2 - 1


#
# Main Loop
#

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_file_path', '-p', type=str, required=True, help='Path to the input file. Current data parsing capability limited.')
    parser.add_argument('--save_file_directory', '-d', type=str, required=True, help='Directory to save the output file')
    parser.add_argument('--skip', type=int, default=500, help='Number of lines to skip (default: 50)')
    parser.add_argument('--seq_length', type=int, default=500, help='Sequence length (default: 500)')
    args = parser.parse_args()

    return args.data_file_path, args.save_file_directory, args.skip, args.seq_length

def path_exists_and_create(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_subdirectories_list(path):
    sub_dirs:list = ['Patient', 'Control']

    dir_list = []
    for sub_dir in sub_dirs:
        dir_list.append(os.path.join(path, sub_dir))
    return dir_list

def main():
    mne.set_log_level('ERROR') 
    file_path, save_file_directory, skip, seq_length = argparse_helper()

    #
    # Data file_path .edf file locating
    #

    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if file.endswith('.edf') and "EC" in file]

    #
    # Save file location validation/creation
    #

    for directory in create_subdirectories_list(save_file_directory):
        path_exists_and_create(directory)

    #
    # Read .edf files
    #

    for file in tqdm(file_list):
        eeg_data = read_eeg(file_path, file)
        # Split into chunks and normalize
        stacked_chunks = []
        for chunk in slide_window(eeg_data, seq_length, skip):
            chunk = normalize(chunk)
            stacked_chunks.append(chunk)
        stacked_chunks = np.stack(stacked_chunks, axis=0)

        #
        # Save Array
        #

        if file.startswith('MDD'):
            sub_directory = 'Patient'
        elif file.startswith('H'):
            sub_directory = 'Control'
        else:
            raise ValueError("File name did not start with H or MDD.")
        np.save(os.path.join(save_file_directory, sub_directory, file[:-4]), stacked_chunks)
        


if __name__ == "__main__":
    main()