import os
import numpy as np
from PIL import Image

npy_folder = r"cwt_loso\Validation\Control"  # Replace with your folder path
output_folder = r"cwt_image\Validation\Control"
os.makedirs(output_folder, exist_ok=True)

# Loop through all .npz files
for npz_file in os.listdir(npy_folder):
    if npz_file.endswith(".npz"):
        print(f"Processing: {npz_file}")
        data = np.load(os.path.join(npy_folder, npz_file))

        # Process each array inside the .npz file
        for key in data.files:  # Access all arrays stored in the .npz
            array = data[key]
            
            # Ensure the array is 2D (single-channel)
            if len(array.shape) != 2:
                print(f"Skipping {key} in {npz_file}: not single-channel (shape: {array.shape}).")
                continue

            # Normalize and scale the array to [0, 255]
            array_normalized = (array - np.min(array)) / (np.max(array) - np.min(array))  # Scale to [0, 1]
            array_scaled = (array_normalized * 255).astype(np.uint8)  # Convert to [0, 255]

            # Convert to an image and save
            image = Image.fromarray(array_scaled)
            output_path = os.path.join(output_folder, f"{npz_file.replace('.npz', '')}_{key}.png")
            image.save(output_path)
            print(f"Saved {output_path}")
