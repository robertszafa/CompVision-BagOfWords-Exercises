"""
CW1-COMP338 - Step 3. Image representation with a histogram of codewords.

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import cv2
import numpy as np
import time
import sys
import re

import helper as hp
import multiprocessing as mp

###########################################################################
# Step 3. Image representation with a histogram of codewords
################################################################################
## Step 3.2
def gen_single_img_histogram(img_descriptors_codebook_pair):
    """
    Generate a histogram of codewords for a single image, which is represented by a list of features.
    """
    img_descriptors, codebook = img_descriptors_codebook_pair

    # Initially, each image will have a count of 0 for each codeword.
    histogram_of_codewords = [0 for _ in range(len(codebook))]

    # Keep track of which descriptor idxs map to which code word.
    descriptor_to_codeword_map = [[] for _ in range(len(codebook))]
    map_descriptor = lambda word_idx, descriptor_idx : \
                      descriptor_to_codeword_map[word_idx].append(descriptor_idx)

    for idx, descriptor in enumerate(img_descriptors):
        # Step 3.1
        closest_cluster_idx = hp.get_idx_of_1_NN(descriptor, codebook, dist_func=hp.euclidean_distance)
        histogram_of_codewords[closest_cluster_idx] += 1

        map_descriptor(closest_cluster_idx, idx)

    return histogram_of_codewords, descriptor_to_codeword_map

## Step 3.4
def normalise_histogram(histogram):
    """
    Given a list {histogram}, where each element {i} represents the frequency of the bin {i},
    return a normalised_histogram where each element {i} is equal to
    (frequency in bin {i}) / (total number of elements in all bins, i.e. L1 norm of the histogram)
    """
    total_sum = np.sum(histogram)
    for i in range(len(histogram)):
        histogram[i] /= total_sum

    return histogram


def gen_histograms(training_descriptors, test_descriptors, training_keypoints, test_keypoints,
                   codebook, hist_file_extension='_histogram.npy', kp_diameter_threshold=30):
    """
    Generate a histogram for all images from the given codebook.
    """

    start_time = time.time()
    # Keep track of indexes of keypoints which mapped to the same codeword. One dictionary of
    # {img_fname: [keypoints]} pairs per codeword.
    map_kps_to_codewords = [dict() for _ in range(len(codebook))]

    for train_or_test in ['Test', 'Training']:
        descriptors_dict = training_descriptors if train_or_test == 'Training' else test_descriptors
        keypoints_dict = training_keypoints if train_or_test == 'Training' else test_keypoints

        for img_class, descriptors_files in descriptors_dict.items():
            # Distribute all img_descriptors fromt this class accross available CPUs.
            with mp.Pool(mp.cpu_count()) as pool:
                # Pack input for map.
                img_descriptors_codebook_pair = \
                    [(descriptors, codebook) for descriptors in descriptors_files.values()]
                img_histograms_descriptor_to_codeword_map_pairs = \
                    pool.map(gen_single_img_histogram, img_descriptors_codebook_pair)

                # Unpack output from map.
                img_histograms, descriptor_to_codeword_maps = [], []
                for hd in img_histograms_descriptor_to_codeword_map_pairs:
                    img_histograms.append(hd[0])
                    descriptor_to_codeword_maps.append(hd[1])

                nor_img_histograms = pool.map(normalise_histogram, img_histograms)

            # Save each image histogram to a seperate file
            for i, img_id in enumerate(descriptors_files.keys()):
                hist_fname = f'{hp.DATASET_DIR}/{train_or_test}/{img_class}/{img_id}{hist_file_extension}'
                hp.save_to_pickle(hist_fname, nor_img_histograms[i])

            for i, img_id in enumerate(descriptors_files.keys()):
                # Use full img path, instead of id, for easier visualisation.
                img_fname = f'{hp.DATASET_DIR}/{train_or_test}/{img_class}/{img_id}.jpg'
                for word_idx, keypoint_idxs_list in enumerate(descriptor_to_codeword_maps[i]):
                    # Get rid of small keypoints.
                    filtered_keypoints = []
                    for kp_idx in keypoint_idxs_list:
                        # We have saved the keypoint as [(kp_x, (kp_y), kp_diameter]
                        # Use the fact that there is a 1:1 mapping between descriptor and kypoint idxs.
                        kp = keypoints_dict[img_class][img_id][kp_idx]
                        if kp[1] > kp_diameter_threshold:
                            filtered_keypoints.append(kp)

                    map_kps_to_codewords[word_idx][img_fname] = filtered_keypoints

            print(f'Finished {train_or_test}/{img_class} in {(time.time() - start_time)/60} minutes.')

    return map_kps_to_codewords


if __name__ == "__main__":
    start_time = time.time()

    codebook = hp.load_pickled_list(hp.CODEBOOK_FILE)
    codebook_small = hp.load_pickled_list(hp.CODEBOOK_SMALL_FILE)

    codebook_euclidean = hp.load_pickled_list(hp.CODEBOOK_EUCLIDEAN_FILE)
    codebook_euclidean_small = hp.load_pickled_list(hp.CODEBOOK_EUCLIDEAN_SMALL_FILE)

    training_descriptors = hp.load_descriptors(test_or_train='Training', merge_in_class=False)
    test_descriptors = hp.load_descriptors(test_or_train='Test', merge_in_class=False)
    # Note that keypoint i of a given image corresponds to descriptor i of that image.
    training_keypoints = hp.load_keypoints(test_or_train='Training', merge_in_class=False)
    test_keypoints = hp.load_keypoints(test_or_train='Test', merge_in_class=False)

    # Generate histograms for the 500- and 20-word codebook where SAD was used as similarity function.
    map_kps_to_codebook = gen_histograms(training_descriptors, test_descriptors,
                                         training_keypoints, test_keypoints,
                                         codebook,
                                         hist_file_extension=hp.HISTOGRAM_FILE_EXT)
    hp.save_to_pickle(hp.MAP_KPS_TO_CODEBOOK_FILE, map_kps_to_codebook)

    map_kps_to_codebook_small = gen_histograms(training_descriptors, test_descriptors,
                                               training_keypoints, test_keypoints,
                                               codebook_small,
                                               hist_file_extension=hp.HISTOGRAM_SMALL_FILE_EXT)
    hp.save_to_pickle(hp.MAP_KPS_TO_CODEBOOK_SMALL_FILE, map_kps_to_codebook_small)

    # # Generate histograms for the 500- and 20-word codebook where euclidean distance was used as similarity function.
    map_kps_to_codebook_euclidean_small = gen_histograms(training_descriptors, test_descriptors,
                                                         training_keypoints, test_keypoints,
                                                         codebook_euclidean_small,
                                                         hist_file_extension=hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT)
    hp.save_to_pickle(hp.MAP_KPS_TO_CODEBOOK_EUCLIDEAN_SMALL_FILE, map_kps_to_codebook_euclidean_small)

    map_kps_to_codebook_euclidean = gen_histograms(training_descriptors, test_descriptors,
                                               training_keypoints, test_keypoints,
                                               codebook_euclidean,
                                               hist_file_extension=hp.HISTOGRAM_EUCLIDEAN_FILE_EXT)
    hp.save_to_pickle(hp.MAP_KPS_TO_CODEBOOK_EUCLIDEAN_FILE, map_kps_to_codebook_euclidean)

    print(f'Finished program in {(time.time() - start_time)/60} minutes.')
