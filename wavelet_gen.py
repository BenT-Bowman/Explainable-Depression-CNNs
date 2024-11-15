import numpy as np
import pywt  # For wavelet transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress tracking
import os
from PIL import Image

def generate_cwt_image(data, wavelet='morl', scales=np.arange(1, 32)):
    coefficients, _ = pywt.cwt(data, scales, wavelet)
    return np.abs(coefficients)

def save_cwt_images(data_list, output_dir="cwt_images"):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for i, data in enumerate(tqdm(data_list, desc="Processing and saving CWT images")):
        # Generate CWT image for the current data sample
        cwt_image = generate_cwt_image(data)

        # Normalize to 0-255 and convert to uint8 to save space
        cwt_image = 255 * (cwt_image / np.max(cwt_image))
        cwt_image = cwt_image.astype(np.uint8)

        # Save the image as a separate .npz file
        np.savez_compressed(os.path.join(output_dir, f"cwt_image_{i}.npz"), image=cwt_image)

    print(f"Saved CWT images to {output_dir} in compressed .npz format")


data_list = np.load('./Leave_one_subject_out/Validation/mdd_control.npy')[:,0,:]
print(data_list.shape)
save_cwt_images(data_list)




