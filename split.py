import os
import numpy as np
from tqdm import tqdm
import argparse
import re
import mne
from random import randint
##
# Utility Functions
##
def sliding_window(full_signal: list, window: int = 1000, features: int = 20, original_length: int = 76800, skip: int = 250):
    """
    Generates overlapping subarrays (windows) from the input signal data, moving by a specified step size.

    Parameters:
    -----------
    full_signal : list
        The individual input signals with dimensions assumed to be [features, length].
    window : int, default=1000
        The length of each window (number of time points in each segment).
    features : int, default=20
        The number of feature channels in each window.
    original_length : int, default=76800
        The original length of the input signal for reference, although not used directly within the function.
    skip : int, default=250
        The step size (in time points) to slide the window by in each iteration, controlling the degree of overlap.

    Yields:
    -------
    np.ndarray
        A window of the signal data with shape (features, window).

    Notes:
    ------
    - If the generated window has a shape other than (features, window), iteration stops.
    - This approach allows efficient extraction of overlapping windows from multichannel time series data.

    Example:
    --------
    ```
    signal = np.random.rand(20, 80000)  # 20 channels, 80000 time points
    windows = list(sliding_window(signal, window=1000, skip=250))
    ```
    """
    for idx in range(0, len(full_signal[-1]), skip):
        signal = full_signal[:20, idx:idx+window] # <- Crazy insidious bug :)
        if signal.shape != (features, window):
            break
        yield signal

def full(data: list, **kwargs) -> np.ndarray:
    """
    Applies a sliding window function to each signal in a list of signals, aggregating all windows into a single array.

    Parameters:
    -----------
    data : list
        A list of input signals, where each signal is expected to be an ndarray with dimensions 
        (features, length).
    **kwargs
        Additional keyword arguments passed to the `sliding_window` function, allowing customization 
        of window size, step size, and other parameters.

    Returns:
    --------
    np.ndarray
        An array containing all windows generated from each signal in the input list. The shape of 
        the output array is (num_windows, features, window_size), where `num_windows` depends on the 
        length and overlap of each signal.

    Notes:
    ------
    - This function uses `tqdm` to display a progress bar, indicating the progress of processing 
      multiple signals.
    - The `sliding_window` function is applied individually to each signal, producing overlapping 
      windows for each one, which are then collected in a list and converted to an ndarray.

    Example:
    --------
    ```
    from tqdm import tqdm
    windows = full(data_list, window=1000, skip=250)
    ```
    """
    temp = []
    for signal in tqdm(data):
        for result_window in sliding_window(signal, **kwargs):
            temp.append(result_window)

    return np.asarray(temp)

def norm(arr: np.ndarray):
    min_vals = np.min(arr, axis=(1, 2))
    max_vals = np.max(arr, axis=(1, 2))

    return (arr - min_vals[:, None, None]) / (max_vals - min_vals)[:, None, None] * 2 - 1

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
    print(len(control), len(patient))

    k = 5
    patient_validation_index = np.random.choice(range(0, len(patient)), size=k, replace=False)
    control_validation_index = np.random.choice(range(0, len(control)), size=k, replace=False)
    patient_validation = [patient[idx] for idx in patient_validation_index]
    control_validation = [control[idx] for idx in control_validation_index]
    patient = [patient[i] for i in range(0, len(patient)) if i not in patient_validation_index]
    control = [control[i] for i in range(0, len(control)) if i not in control_validation_index]

    print(len(control), control[0].shape, len(patient), patient[0].shape)
    print(len(control_validation), control_validation[0].shape, len(patient_validation), patient_validation[0].shape)

    
    patient = norm(full(patient, skip=skip, window=seq_length))
    control = norm(full(control, skip=skip, window=seq_length))
    patient_validation = norm(full(patient_validation, skip=skip, window=seq_length))
    control_validation = norm(full(control_validation, skip=skip, window=seq_length))


    np.save(fr'{save_file_directory}/Main/mdd_control', control )
    np.save(fr'{save_file_directory}/Main/mdd_patient', patient )
    np.save(fr'{save_file_directory}/Validation/mdd_control', control_validation )
    np.save(fr'{save_file_directory}/Validation/mdd_patient', patient_validation )

    print("Finished: ",control.shape)
    print("Finished: ",patient.shape)
    print("Finished: ",control_validation.shape)
    print("Finished: ",patient_validation.shape)





if __name__ == "__main__":
    main()