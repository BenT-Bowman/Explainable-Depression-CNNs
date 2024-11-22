from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import seaborn as sns
from random import choice

files = os.listdir(r'cwt_loso\Main\Control')
eeg_data = np.load(fr'cwt_loso\Main\Control\{choice(files)}')['image']

print(eeg_data.shape)

plt.imshow(eeg_data, cmap='viridis')
plt.colorbar()
plt.show()