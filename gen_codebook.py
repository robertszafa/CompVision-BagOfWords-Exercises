"""
CW1-COMP338 - Step 2. Dictionary generation

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import random, time
import cv2
import numpy as np
import multiprocessing as mp

import helper as hp

################################################################################
# Step 2. Dictionary generation
################################################################################
def find_closest_neighbour_idx(neighbours_candidate_dist_func):
    """
    Given a tuple of (neighbours, candidate, dist_func),
    return the index of the 1-kNN of candidate in neighbours.
    """
    neighbours, candidate, dist_func = neighbours_candidate_dist_func

    return hp.get_idx_of_1_NN(candidate, neighbours, dist_func=dist_func)

def gen_codebook(feature_descriptors, fname, dist_func=hp.sad, num_words=500, max_iter=10):
    """
    Cluser feuture_descriptors into {num_words} clusters.
    The generated codebook is saved to a file {fname} after each each iteration.
    """
    start_time = time.time()

    # Initialise. Randomly choose num_words feature descriptors as cluster centres.
    codebook = []
    random_idxs = np.random.choice(len(feature_descriptors), num_words)
    for i in random_idxs:
        codebook.append(feature_descriptors[i])

    # Do clustering while there are any changes in any cluster centre, but not more than max_iter.
    for iteration in range(1, max_iter+1):
        # Find the indexes of the nearest cluster for each descriptor.
        # This step is easily parallelizable.
        with mp.Pool(mp.cpu_count()) as pool:
            # pool.map takes one argument.
            map_input = [(codebook, descriptor, dist_func) for descriptor in feature_descriptors]
            closest_cluster_idxs = pool.map(find_closest_neighbour_idx, map_input)

        # Collect all the descrtiptors mapped to the same keyword from the calculated indexes.
        cluster_vectors_map = [[word] for word in codebook]
        for i in range(len(closest_cluster_idxs)):
            # pool.map results are ordered,
            # i.e. the output closest_cluster_idxs[i] corresponds to the descriptor[i]
            cluster_vectors_map[closest_cluster_idxs[i]].append(feature_descriptors[i])

        # Calculate new cluster centers. This is also easily parallelizable.
        with mp.Pool(mp.cpu_count()) as pool:
            new_centers = pool.map(hp.mean, cluster_vectors_map)

        # Compare to previous iteration codebook
        diff = abs(np.array(codebook) - np.array(new_centers))

        # Assign new centers.
        for i in range(len(codebook)):
            codebook[i] = new_centers[i]

        hp.save_to_pickle(fname, codebook)
        print(f'Finished iteration {iteration} at minute {(time.time() - start_time)/60}.')

        # Stop if the improvements are very small.
        delta = 1.0
        if np.all(diff < delta):
            break

    return codebook

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    start_time = time.time()

    # Merge the descriptors from one class into a single list.
    # training_descriptors will hold ['class_name': descriptors_list] pairs
    training_descriptors = hp.load_descriptors(test_or_train='Training', merge_in_class=True)

    # A single list for all feature descriptors from all classes.
    all_descriptors = []
    for descriptors in training_descriptors.values():
        all_descriptors += descriptors

    codebook = gen_codebook(all_descriptors, hp.CODEBOOK_FILE, dist_func=hp.sad, num_words=500)
    codebook_small = gen_codebook(all_descriptors, hp.CODEBOOK_SMALL_FILE, dist_func=hp.sad, num_words=20)

    codebook_euclidean = gen_codebook(all_descriptors, hp.CODEBOOK_EUCLIDEAN_FILE,
                                      dist_func=hp.euclidean_distance, num_words=500)
    codebook_small_euclidean = gen_codebook(all_descriptors, hp.CODEBOOK_EUCLIDEAN_SMALL_FILE,
                                            dist_func=hp.euclidean_distance, num_words=20)

    print(f'Finished program in {(time.time() - start_time)/60} minutes.')
