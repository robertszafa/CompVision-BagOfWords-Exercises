
import cv2
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
def find_closest_neighbour_idx(neighbours, candidate):
    closest_idx = 0
    curr_dist = np.sum(abs(neighbours[closest_idx] - candidate))

    for i in range(len(neighbours)):
        if np.sum(abs(neighbours[i] - candidate)) < curr_dist:
            closest_idx = i
            curr_dist = np.sum(abs(neighbours[closest_idx] - candidate))

    return closest_idx

def gen_dictionary(feature_descriptors, num_words=500):
    codebook = []

    # Initialise. Choose the first 500 features as cluster centres.
    for i in range(num_words):
        descriptor = feature_descriptors[i]
        codebook.append([descriptor, 1]) # Cluster {center, population} pairs.

    for descriptor in feature_descriptors[num_words:]:
        cluster_centers = [codeword_pair[0] for codeword_pair in codebook]
        closest_cluster_idx = find_closest_neighbour_idx(cluster_centers, descriptor)

        # Update cluster center and increase count
        new_center = (codebook[closest_cluster_idx][0] + descriptor) / 2
        codebook[closest_cluster_idx][0] = new_center
        codebook[closest_cluster_idx][1] += 1

    return codebook





################################################################################
# Helper functions
################################################################################
def read_descriptors(flat=True):
    training_descriptors = {}
    test_descriptors = {}

    for class_name in CLASSES:
        with open(f'{DATASET_DIR}/training_descriptors/{class_name}.npy', 'rb') as f:
            training_descriptors[class_name] = np.load(f, allow_pickle=True)

        with open(f'{DATASET_DIR}/test_descriptors/{class_name}.npy', 'rb') as f:
            test_descriptors[class_name] = np.load(f, allow_pickle=True)

        if flat:
            # Put all descriptors from all images into a single list.
            tmp = training_descriptors[class_name]
            training_descriptors[class_name] = []
            for img in tmp:
                for descriptor in img:
                    training_descriptors[class_name].append(descriptor)

            tmp = test_descriptors[class_name]
            test_descriptors[class_name] = []
            for img in tmp:
                for descriptor in img:
                    test_descriptors[class_name].append(descriptor)


    return training_descriptors, test_descriptors

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    # Read descriptors from binary files.
    training_descriptors, test_descriptors = read_descriptors()

    # Generate a codebook for each class.
    codebook = {}
    for img_class, descriptors in training_descriptors.items():
        codebook[img_class] = gen_dictionary(descriptors)

    print(len(codebook))
    print(len(codebook[0]))
    print(codebook)

