import numpy as np
from random import randint
import matplotlib.pyplot as plt

def show_eeg(eeg):
    for channel in eeg:
        plt.plot(channel)
    plt.show()

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    dataset = np.load(r'train_data\leave_one_subject_out_CV\Control\H S30 EC.npy')
    # print(dataset.shape)
    length = dataset.shape[0]

    eeg = dataset[randint(0, length-1), :]
    show_eeg(eeg)