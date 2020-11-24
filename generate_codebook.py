import random, time, re, os
import cv2
import numpy as np

from helper import load_descriptors, save_to_pickle, euclidean_distance
from helper import DATASET_DIR, CLASSES, CODEBOOK_FILE_TRAIN, SMALL_CODEBOOK_FILE_TRAIN


################################################################################
# Step 2. Dictionary generation
################################################################################
def find_closest_neighbour_idx(neighbours, candidate):
    closest_idx = 0
    curr_dist = euclidean_distance(candidate, neighbours[closest_idx])

    for i in range(len(neighbours)):
        this_dist = euclidean_distance(candidate, neighbours[i])

        if this_dist < curr_dist:
            closest_idx = i
            curr_dist = this_dist

    return closest_idx

def gen_dictionary(feature_descriptors, num_words=500):
    start_time = time.time()

    # Initialise. Randomly choose num_words feature descriptors as cluster centres.
    codebook = []
    random_idxs = np.random.choice(len(feature_descriptors), num_words)
    for i in random_idxs:
        descriptor = feature_descriptors[i]
        codebook.append(descriptor)

    # Do, while there were any changes in any cluster.
    do_next_iter = True
    max_iter = 100
    iteration = 0
    while do_next_iter and iteration < max_iter:
        iteration += 1
        do_next_iter = False

        for descriptor in feature_descriptors:
            closest_cluster_idx = find_closest_neighbour_idx(codebook, descriptor)

            # Update cluster center and increase count.
            new_center = (codebook[closest_cluster_idx] + descriptor) / 2

            # Stop when the improvements become negligable.
            delta_for_change = 10
            if do_next_iter or euclidean_distance(codebook[closest_cluster_idx], new_center) > delta_for_change:
                codebook[closest_cluster_idx] = new_center
                do_next_iter = True

        print(f'Finished iteration {iteration} at minute {(time.time() - start_time)/60}.')

    return codebook

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    start_time = time.time()

    # Merge the descriptors from one class into a single list.
    # training_descriptors will hold ['class_name': descriptors_list] pairs
    training_descriptors = load_descriptors(test_or_train='Training', merge_in_class=True)

    # A single list for all feature descriptors from all classes.
    # Pick the same number of descriptors from each class to prevent bias in the code book.
    min_len_descriptors = min(map(len, training_descriptors.values()))
    all_descriptors = []
    for descriptors in training_descriptors.values():
        all_descriptors += random.sample(descriptors, min_len_descriptors)

    # Generate one codebook for all classes.
    codebook = gen_dictionary(descriptors)
    save_to_pickle(CODEBOOK_FILE_TRAIN, codebook)

    small_codebook = gen_dictionary(descriptors, num_words=20)
    save_to_pickle(SMALL_CODEBOOK_FILE_TRAIN, small_codebook)


    print(f'Finished program in {(time.time() - start_time)/60} minutes.')
