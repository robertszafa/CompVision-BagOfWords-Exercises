import random, time, re, os
import cv2
import numpy as np

from helper import load_descriptors, save_to_pickle, euclidean_distance, mean
from helper import DATASET_DIR, CLASSES, CODEBOOK_FILE_TRAIN, SMALL_CODEBOOK_FILE_TRAIN, UNLIMITED_CODEBOOK_FILE_TRAIN, UNLIMITED_SMALL_CODEBOOK_FILE_TRAIN

# Speed up k-NN
import multiprocessing as mp


################################################################################
# Step 2. Dictionary generation
################################################################################
def find_closest_neighbour_idx(neighbours_candidate_pair):
    """
    Given a tuple of (neighbours, candidate), return the index of the 1-kNN of candidate in neighbours.
    """
    neighbours, candidate = neighbours_candidate_pair

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

    # Get rid of the selected descriptors.
    feature_descriptors = [feature_descriptors[i] for i in range(len(feature_descriptors)) if i not in random_idxs]

    # Do, while there were any changes in any cluster.
    do_next_iter = True
    max_iter = 10
    iteration = 0
    while do_next_iter and iteration < max_iter:
        iteration += 1
        do_next_iter = False

        # Find the indexes of the nearest cluster for each descriptor.
        # This step is easily parallelizable.
        with mp.Pool(mp.cpu_count()) as pool:
            # pool.map takes one argument, so we have to zip codebook and descriptor together.
            # Note that we need unique copies of the codebook since each process might execute
            # on different CPUs.
            neighbours_candidate_pair = [(codebook, descriptor) for descriptor in descriptors]
            closest_cluster_idxs = pool.map(find_closest_neighbour_idx, neighbours_candidate_pair)

        # Collect all the descrtiptors mapped to the same keyword from the calculated indexes.
        # pool.map results are ordered,
        # i.e. the output closest_cluster_idxs[i] corresponds to the descriptor[i]
        cluster_vectors_map = [[word] for word in codebook]
        for i in closest_cluster_idxs:
            cluster_vectors_map[closest_cluster_idxs[i]].append(descriptors[i])

        # Calculate new cluster centers. This is also easily parallelizable.
        with mp.Pool(mp.cpu_count()) as pool:
            new_centers = pool.map(mean, cluster_vectors_map)

        # Stop if there are no more improvements to be made.
        do_next_iter = not np.all(codebook == new_centers)

        # Assign new centers.
        for i in range(len(codebook)):
            codebook[i] = new_centers[i]

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
