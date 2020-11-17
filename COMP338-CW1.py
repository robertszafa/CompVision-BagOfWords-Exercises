
import cv2
import os
import numpy as np

from SIFT import extract_SIFT_features


################################################################################
# Constants
################################################################################
DATASET_DIRECTORY = 'COMP338_Assignment1_Dataset'
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
# Helper functions
################################################################################
def read_images_in_directory(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)

    return images

def get_tarining_images():
    training_images = []
    for name in CLASSES:
        training_images += read_images_in_directory(DATASET_DIRECTORY + '/Training/' + name)

    return training_images


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    training_images = get_tarining_images()