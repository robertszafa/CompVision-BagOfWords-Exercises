
import cv2
import os
import numpy as np


################################################################################
# Constants
################################################################################
DATASET_DIR = 'COMP338_Assignment1_Dataset'
CLASSES = [
    'airplanes',
    'cars',
    'dog',
    'faces',
    'keyboard',
]


################################################################################
# Step 2. Dictionary generation
################################################################################
def gen_dictionary(images):
    pass


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    # Read descriptors from a binary file.
    with open(f'{DATASET_DIR}/test_descriptors/cars.npy', 'rb') as f:
        descriptors = np.load(f, allow_pickle=True)

    print(descriptors)
