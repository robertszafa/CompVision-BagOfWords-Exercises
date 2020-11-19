
import cv2
import numpy as np
import time

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

    # Initialise. Randomly choose num_words features as cluster centres.
    random_idxs = np.random.choice(len(feature_descriptors), num_words)
    for i in random_idxs:
        # :TODO: Remove from the list here?
        # descriptor = feature_descriptors.pop(i)
        descriptor = feature_descriptors[i]
        codebook.append(descriptor)

    # Do: while therw were any changes in any cluster.
    no_change = False
    max_iter = 10
    iteration = 0
    while not no_change and iteration < max_iter:
        iteration += 1
        no_change = True
        for descriptor in feature_descriptors:
            closest_cluster_idx = find_closest_neighbour_idx(codebook, descriptor)

            # Update cluster center and increase count.
            new_center = (codebook[closest_cluster_idx] + descriptor) / 2

            # Stop when the improvements become very small.
            delta_for_change = 10
            if np.sum(abs(codebook[closest_cluster_idx] - new_center) > delta_for_change):
                codebook[closest_cluster_idx] = new_center
                no_change = False

        print('iter++')
    print('end while')

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
    start_time = time.time()

    # Read descriptors from binary files.
    training_descriptors, test_descriptors = read_descriptors()

    # Generate a codebook for each class.
    codebook = {}
    for img_class, descriptors in training_descriptors.items():
        codebook[img_class] = gen_dictionary(descriptors)

    with open(CODEBOOK_FILE, 'wb') as f:
        np.save(f, codebook)


    print(f'Finished program in {(time.time() - start_time)/60} minutes.')
