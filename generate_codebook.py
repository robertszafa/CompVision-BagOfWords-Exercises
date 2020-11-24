import random, time, re, os
import cv2
import numpy as np

from helper import load_descriptors, save_to_pickle, euclidean_distance, mean
from helper import DATASET_DIR, CLASSES, CODEBOOK_FILE_TRAIN, SMALL_CODEBOOK_FILE_TRAIN, UNLIMITED_CODEBOOK_FILE_TRAIN, UNLIMITED_SMALL_CODEBOOK_FILE_TRAIN


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

def gen_codebook(feature_descriptors, fname, num_words=500):
    start_time = time.time()

    # Initialise. Randomly choose num_words feature descriptors as cluster centres.
    codebook = []
    random_idxs = np.random.choice(len(feature_descriptors), num_words)
    for i in random_idxs:
        codebook.append(feature_descriptors[i])

    # Get rid of the selected centers.
    feature_descriptors = [feature_descriptors[i] for i in range(len(feature_descriptors)) if i not in random_idxs]

    # Do, while there were any changes in any cluster.
    do_next_iter = True
    max_iter = 10
    iteration = 0
    while do_next_iter and iteration < max_iter:
        iteration += 1
        do_next_iter = False

        new_centers = [[word] for word in codebook]
        for i, descriptor in enumerate(feature_descriptors):
            closest_cluster_idx = find_closest_neighbour_idx(codebook, descriptor)

            new_centers[closest_cluster_idx].append(descriptor)

        # Stop when there are no improvements.
        for assigned_descriptors in new_centers:
            new_codeword = mean(assigned_descriptors)
            if new_codeword != codebook[i]:
                codebook[i] = new_codeword
                do_next_iter = True

        print(f'Finished iteration {iteration} at minute {(time.time() - start_time)/60}.')
        save_to_pickle(fname, codebook)

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
    capped_descriptors = []
    for descriptors in training_descriptors.values():
        capped_descriptors += random.sample(descriptors, min_len_descriptors)
        all_descriptors += descriptors

    gen_codebook(capped_descriptors, fname=CODEBOOK_FILE_TRAIN, num_words=500)
    gen_codebook(capped_descriptors, fname=SMALL_CODEBOOK_FILE_TRAIN, num_words=20)

    gen_codebook(all_descriptors, fname=UNLIMITED_CODEBOOK_FILE_TRAIN, num_words=500)
    gen_codebook(all_descriptors, fname=UNLIMITED_SMALL_CODEBOOK_FILE_TRAIN, num_words=20)

    print(f'Finished program in {(time.time() - start_time)/60} minutes.')
