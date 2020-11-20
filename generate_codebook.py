from typing import Dict, List
import cv2
import numpy as np
import time
import re
import os

################################################################################
# Constants
################################################################################
DATASET_DIR = 'COMP338_Assignment1_Dataset'
CODEBOOK_FILE_TRAIN = f'{DATASET_DIR}/Training/codebook.npy'
CODEBOOK_FILE_TEST = f'{DATASET_DIR}/Test/codebook.npy'
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

    # Do, while there were any changes in any cluster.
    # Use the L1 norm to check for closeness.
    # l1_norm = lambda v : np.sum(abs(v))
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

            # Stop when the improvements become negligable.
            delta_for_change = 10
            if np.any(abs(codebook[closest_cluster_idx] - new_center) > delta_for_change):
                codebook[closest_cluster_idx] = new_center
                no_change = False

    return codebook


################################################################################
# Helper functions
################################################################################
def load_np_pickles_in_directory(path, regex=r'.*.(npy|npc)'):
    result = {}
    for filename in os.listdir(path):
        if re.match(regex, filename):
            result[filename] = np.load(path + filename, allow_pickle=True)

    return result

def load_descriptors(test_or_train, merge_in_class=False):
    """
    Read the descriptors from the {test_or_train} dataset.
    Return a dictionary with the class names as keys.
    If {merge_in_class} is True, then a single class will have a list of all descriptors as the value.
    Otherwise, it will have a list of dictionaries as values, where the dictionaries have
    the individual img filename as key and their list of descriptors as values.
    """
    descriptors = {}
    for class_name in CLASSES:
        match_descriptors = r'.*_descriptors' + re.escape('.npy')
        load_from = f'{DATASET_DIR}/{test_or_train}/{class_name}/'
        descriptors_dict = load_np_pickles_in_directory(load_from, match_descriptors)

        if merge_in_class:
            # Merge all img descriptors from tge same class into one list.
            # We ignore the individual img file names here.
            class_descriptors = []
            for img_descriptors in descriptors_dict.values():
                for d in img_descriptors:
                    class_descriptors.append(d)

            descriptors[class_name] = class_descriptors
        else:
            descriptors[class_name] = descriptors_dict

    return descriptors

def load_keypoints(test_or_train, merge_in_class=False):
    """
    Read the keypoints from the {test_or_train} dataset.
    Return a dictionary with the class names as keys.
    If {merge_in_class} is True, then a single class will have a list of all keypoints as the value.
    Otherwise, it will have a list of dictionaries as values, where the dictionaries have
    the individual img filename as key and their list of keypoints as values.
    """
    keypoints = {}
    for class_name in CLASSES:
        match_keypoints = r'.*_keypoints' + re.escape('.npy')
        load_from = f'{DATASET_DIR}/{test_or_train}/{class_name}/'
        keypoints_dict = load_np_pickles_in_directory(load_from, match_keypoints)

        if merge_in_class:
            # Merge all img keypoints from tge same class into one list.
            # We ignore the individual img file names here.
            class_keypoints = []
            for img_keypoints in keypoints_dict.values():
                for d in img_keypoints:
                    class_keypoints.append(d)

            keypoints[class_name] = class_keypoints
        else:
            keypoints[class_name] = keypoints_dict

    return keypoints


def read_codebook() -> List[Dict[str, List[int]]]:
    load_training_codebook = np.load(CODEBOOK_FILE_TRAIN, allow_pickle=True)
    load_test_codebook = np.load(CODEBOOK_FILE_TEST, allow_pickle=True)
    return load_training_codebook.tolist(), load_test_codebook.tolist()


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    start_time = time.time()

    # Merge the descriptors from one class into a single list.
    # training_descriptors will hold ['class_name': descriptors_list] pairs
    training_descriptors = load_descriptors(test_or_train='Training', merge_in_class=True)

    # Generate a codebook for each class.
    codebook = {}
    for img_class, descriptors in training_descriptors.items():
        codebook[img_class] = gen_dictionary(descriptors)
        print(f'Finished {img_class} at minute {(time.time() - start_time)/60}.')

    with open(CODEBOOK_FILE_TRAIN, 'wb') as f:
        np.save(f, codebook)


    print(f'Finished program in {(time.time() - start_time)/60} minutes.')

