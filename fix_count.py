""" Yes I'm lazy, sue me"""

import os
import shutil
import random

def move_excess_samples(folder_path, target_count):
    for class_name in os.listdir(folder_path):
        # print("HELLO")
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            print("HELLO")
            samples = os.listdir(class_folder)
            if len(samples) > target_count:
                excess_samples = random.sample(samples, len(samples) - target_count)
                
                # Create the "extra_samples" folder
                extra_samples_folder = os.path.join(folder_path, "extra_samples", class_name)
                os.makedirs(extra_samples_folder, exist_ok=True)
                
                for sample in excess_samples:
                    shutil.move(os.path.join(class_folder, sample), extra_samples_folder)

# Example usage
move_excess_samples("cwt_data", target_count=9130)
