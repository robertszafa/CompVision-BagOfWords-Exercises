import cv2
import numpy as np
import time

from generate_codebook import read_descriptors

################################################################################
# Constants
################################################################################
DATASET_DIR = 'COMP338_Assignment1_Dataset'
CODEBOOK_FILE = f'{DATASET_DIR}/Training/codebook.npy'
CLASSES = [
    'airplanes',
    'cars',
    'dog',
    'faces',
    'keyboard',
]

################################################################################
# Step 2. Image representation with a histogram of codewords
################################################################################
# Step 3.1
def euclidean_distance(vec1, vec2):
    diff = vec1 - vec2

    dist = 0
    for p in diff:
        # Euclidean distance of 2 vectors is: np.sqrt(p.x ** 2 + p.y**2), but our p has only one
        # value, this is the same as taking the absolute.
        dist += abs(p)

    return dist

# Step 3.1
def find_nearest_cluster_idx(descriptor, clusters):
    min_idx = 0
    min_dist = euclidean_distance(descriptor, clusters[min_idx])

    for i in range(1, len(clusters)):
        if euclidean_distance(descriptor, clusters[i]) < min_dist:
            min_dist = euclidean_distance(descriptor, clusters[i])
            min_idx = i

    return min_idx

# Step 3.2
def gen_histogram_of_codewords(img_descriptors, img_class_codebook):
    """
    Generate a histogram of codewords for a single image, which is represented by a list of features.
    """
    # Initially, each image will have a count of 0 for each codeword.
    histogram_of_codewords = [0 for _ in range(len(img_class_codebook))]

    for descriptor in img_descriptors:
        closest_cluster_idx = find_nearest_cluster_idx(descriptor, img_class_codebook)
        histogram_of_codewords[closest_cluster_idx] += 1

    return histogram_of_codewords

# Step 3.3
def l1_norm(vec):
    return np.sum(abs(vec))



def read_codebook():
    # Read codebook.
    with open(CODEBOOK_FILE, 'rb') as f:
        tmp_codebook = np.load(f, allow_pickle=True)

    # numpy wraps the python dictionary into an object array. Transform it back into a python dictionary.
    codebook = {}
    for img_class in CLASSES:
        codebook[img_class] = tmp_codebook[()][img_class]

    return codebook



if __name__ == "__main__":
    codebook = read_codebook()

    # Get image descriptors, where each image has its own list of descriptors.
    training_descriptors, test_descriptors = read_descriptors(flat=False)

    # For each class, there will be a list of images. Each image will have a histogram, which is
    # a list itself where histogram[idx] is the number of occurances of codeword[idx]
    training_histogram_of_codewords = {img_class: [] for img_class in CLASSES}
    test_histogram_of_codewords = {img_class: [] for img_class in CLASSES}
    for img_class in CLASSES:
        # Training images
        for img_descriptors in training_descriptors[img_class]:
            img_histogram = gen_histogram_of_codewords(img_descriptors, codebook[img_class])
            training_histogram_of_codewords[img_class].append(img_histogram)

        # Same for Test images.
        for img_descriptors in test_descriptors[img_class]:
            img_histogram = gen_histogram_of_codewords(img_descriptors, codebook[img_class])
            test_histogram_of_codewords[img_class].append(img_histogram)

    with open('training_histogram_of_codewords.npu', 'wb') as f:
        np.save(f, training_histogram_of_codewords)

    with open('test_histogram_of_codewords.npu', 'wb') as f:
        np.save(f, test_histogram_of_codewords)
