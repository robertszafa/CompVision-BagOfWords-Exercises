import cv2
import numpy as np
import time
import sys
import re

import helper as hp
import multiprocessing as mp

import scipy.cluster.vq as vq

###########################################################################
# Step 3. Image representation with a histogram of codewords
################################################################################
## Step 3.1
def find_nearest_cluster_idx(descriptor, clusters):
    min_idx = 0
    min_dist = hp.euclidean_distance(descriptor, clusters[min_idx])

    for i in range(1, len(clusters)):
        this_dist = hp.euclidean_distance(descriptor, clusters[i])

        if this_dist < min_dist:
            min_dist = this_dist
            min_idx = i

    return min_idx

## Step 3.2
def gen_histogram_of_codewords(img_descriptors_codebook_pair):
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
        closest_cluster_idx = find_nearest_cluster_idx(descriptor, codebook)
        histogram_of_codewords[closest_cluster_idx] += 1

        map_descriptor(closest_cluster_idx, idx)

    return histogram_of_codewords, descriptor_to_codeword_map

## Step 3.3
def draw_keypoint(img_fname, kp_x, kp_y, kp_diameter, title=''):
    img = cv2.imread(img_fname)
    radius = kp_diameter//2
    kp = img[kp_x-radius : kp_x+radius, kp_y-radius : kp_y+radius]

    cv2.imshow(title, kp)
    cv2.waitKey(0)

def visualize_similar_patches(map_kps_to_codewords):
    for word_idx, img_fname_keypoints_pairs in enumerate(map_kps_to_codewords):
        for img_fname, kps in img_fname_keypoints_pairs.items():
            for kp in kps:
                title = title=f'{img_fname.split(hp.DATASET_DIR)[1]} --> codeword_{word_idx}'
                draw_keypoint(img_fname, int(kp[0][0]), int(kp[0][1]), int(kp[1]), title)

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
                   codebook, hist_file_extension='_histogram.npy', kp_diameter_threshold = 30):
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
            for img_id, descriptors in descriptors_files.items():
                hist = computeHistograms(codebook, descriptors)
                hist_fname = f'{hp.DATASET_DIR}/{train_or_test}/{img_class}/{img_id}{hist_file_extension}'
                hp.save_to_pickle(hist_fname, hist)


            # with mp.Pool(mp.cpu_count()) as pool:
            #     # Pack input for map.
            #     img_descriptors_codebook_pair = \
            #         [(descriptors, codebook) for descriptors in descriptors_files.values()]
            #     img_histograms_descriptor_to_codeword_map_pairs = \
            #         pool.map(gen_histogram_of_codewords, img_descriptors_codebook_pair)

            #     # Unpack output from map.
            #     img_histograms, descriptor_to_codeword_maps = [], []
            #     for hd in img_histograms_descriptor_to_codeword_map_pairs:
            #         img_histograms.append(hd[0])
            #         descriptor_to_codeword_maps.append(hd[1])

            #     nor_img_histograms = pool.map(normalise_histogram, img_histograms)

            # # Save each image histogram to a seperate file
            # for i, img_id in enumerate(descriptors_files.keys()):
            #     hist_fname = f'{hp.DATASET_DIR}/{train_or_test}/{img_class}/{img_id}{hist_file_extension}'
            #     hp.save_to_pickle(hist_fname, nor_img_histograms[i])

            # for i, img_id in enumerate(descriptors_files.keys()):
            #     # Use full img path, instead of id, for easier visualisation.
            #     img_fname = f'{hp.DATASET_DIR}/{train_or_test}/{img_class}/{img_id}.jpg'
            #     for word_idx, keypoint_idxs_list in enumerate(descriptor_to_codeword_maps[i]):
            #         # Get rid of small keypoints.
            #         filtered_keypoints = []
            #         for kp_idx in keypoint_idxs_list:
            #             # We have saved the keypoint as [(kp_x, (kp_y), kp_diameter]
            #             # Use the fact that there is a 1:1 mapping between descriptor and kypoint idxs.
            #             kp = keypoints_dict[img_class][img_id][kp_idx]
            #             if kp[1] > kp_diameter_threshold:
            #                 filtered_keypoints.append(kp)

            #         map_kps_to_codewords[word_idx][img_fname] = filtered_keypoints

            print(f'Finished {train_or_test}/{img_class} in {(time.time() - start_time)/60} minutes.')

    return map_kps_to_codewords


def computeHistograms(codebook, descriptors):
    code, _ = vq.vq(descriptors, codebook)
    histogram_of_words, _ = np.histogram(code, bins=len(codebook), normed=True)
    return histogram_of_words


if __name__ == "__main__":
    start_time = time.time()

    codebook = hp.load_pickled_list(hp.CODEBOOK_FILE_TRAIN)
    # codebook_small = hp.load_pickled_list(hp.SMALL_CODEBOOK_FILE_TRAIN)
    # sad_codebook = hp.load_pickled_list(hp.SAD_CODEBOOK_FILE_TRAIN)
    # sad_codebook_small = hp.load_pickled_list(hp.SAD_SMALL_CODEBOOK_FILE_TRAIN)

    # codebook_unlimited = hp.load_pickled_list(hp.UNLIMITED_CODEBOOK_FILE_TRAIN)
    # codebook_small_unlimited = hp.load_pickled_list(hp.UNLIMITED_SMALL_CODEBOOK_FILE_TRAIN)
    # sad_codebook_unlimited = hp.load_pickled_list(hp.SAD_UNLIMITED_CODEBOOK_FILE_TRAIN)
    # sad_codebook_small_unlimited = hp.load_pickled_list(hp.SAD_UNLIMITED_SMALL_CODEBOOK_FILE_TRAIN)

    training_descriptors = hp.load_descriptors(test_or_train='Training', merge_in_class=False)
    test_descriptors = hp.load_descriptors(test_or_train='Test', merge_in_class=False)
    # Note that keypoint i of a given image corresponds to descriptor i of that image.
    training_keypoints = hp.load_keypoints(test_or_train='Training', merge_in_class=False)
    test_keypoints = hp.load_keypoints(test_or_train='Test', merge_in_class=False)

    map_kps_to_codebook = gen_histograms(training_descriptors, test_descriptors,
                                         training_keypoints, test_keypoints,
                                         codebook, hist_file_extension='_histogram2.npy')
    # hp.save_to_pickle(hp.MAP_KPS_TO_CODEBOOK_FILE, map_kps_to_codebook)

    # map_kps_to_codebook_small = gen_histograms(training_descriptors, test_descriptors,
    #                                            training_keypoints, test_keypoints,
    #                                            codebook_small, hist_file_extension='_histogram_small.npy')
    # hp.save_to_pickle(hp.MAP_KPS_TO_SMALL_CODEBOOK_FILE, map_kps_to_codebook)

    # map_kps_to_sad_codebook = gen_histograms(training_descriptors, test_descriptors,
    #                                            training_keypoints, test_keypoints,
    #                                            sad_codebook, hist_file_extension='_histogram_sad.npy')
    # map_kps_to_sad_codebook_small = gen_histograms(training_descriptors, test_descriptors,
    #                                            training_keypoints, test_keypoints,
    #                                            sad_codebook_small, hist_file_extension='_histogram_sad_small.npy')


    map_kps_to_codebook_unlimited = gen_histograms(training_descriptors, test_descriptors,
                                               training_keypoints, test_keypoints,
                                               codebook_unlimited, hist_file_extension='_histogram_unlimited.npy')
    hp.save_to_pickle(hp.UNLIMITED_MAP_KPS_TO_CODEBOOK_FILE, map_kps_to_codebook_unlimited)

    map_kps_to_codebook_small_unlimited = gen_histograms(training_descriptors, test_descriptors,
                                               training_keypoints, test_keypoints,
                                               codebook_small_unlimited, hist_file_extension='_histogram_unlimited_small.npy')
    hp.save_to_pickle(hp.UNLIMITED_MAP_KPS_TO_SMALL_CODEBOOK_FILE, map_kps_to_codebook_small_unlimited)

    map_kps_to_sad_codebook_unlimited = gen_histograms(training_descriptors, test_descriptors,
                                               training_keypoints, test_keypoints,
                                               sad_codebook_unlimited, hist_file_extension='_histogram_sad_unlimited.npy')
    hp.save_to_pickle(hp.UNLIMITED_MAP_KPS_TO_SAD_CODEBOOK_FILE, map_kps_to_sad_codebook_unlimited)

    map_kps_to_sad_codebook_small_unlimited = gen_histograms(training_descriptors, test_descriptors,
                                               training_keypoints, test_keypoints,
                                               sad_codebook_small_unlimited, hist_file_extension='_histogram_sad_unlimited_small.npy')
    hp.save_to_pickle(hp.UNLIMITED_MAP_KPS_TO_SMALL_SAD_CODEBOOK_FILE, map_kps_to_sad_codebook_small_unlimited)



    #
    # Step 3.3 Visualize some image patches that are assigned to the same codeword.
    #
    map_kps_to_codebook = hp.load_pickled_list(hp.MAP_KPS_TO_CODEBOOK_FILE)
    # Put most matched codewords and the corresponding keypoints to the front.
    # Note that we don't care towhich specific codeword given keypoints match, we just care about
    # the fact that they match to the same one.
    num_of_kps = lambda d : np.sum([len(v) for v in d.values()])
    map_kps_to_codebook.sort(key=num_of_kps, reverse=True)

    visualize_similar_patches(map_kps_to_codebook)


    print(f'Finished program in {(time.time() - start_time)/60} minutes.')


    training_descriptors = hp.load_descriptors(test_or_train='Training', merge_in_class=False)
    test_descriptors = hp.load_descriptors(test_or_train='Test', merge_in_class=False)
    training_keypoints = hp.load_keypoints(test_or_train='Training', merge_in_class=False)
    test_keypoints = hp.load_keypoints(test_or_train='Test', merge_in_class=False)

    # map_kps_to_codebook = gen_histograms(training_descriptors, test_descriptors,
    #                                      training_keypoints, test_keypoints,
    #                                      codebook, hist_file_extension='_histogram.npy')
    # hp.save_to_pickle(hp.MAP_KPS_TO_CODEBOOK_FILE, map_kps_to_codebook)

    # map_kps_to_codebook_small = gen_histograms(training_descriptors, test_descriptors,
    #                                            training_keypoints, test_keypoints,
    #                                            codebook_small, hist_file_extension='_histogram_small.npy')
    # hp.save_to_pickle(hp.MAP_KPS_TO_SMALL_CODEBOOK_FILE, map_kps_to_codebook_small)

    # map_kps_to_sad_codebook = gen_histograms(training_descriptors, test_descriptors,
    #                                            training_keypoints, test_keypoints,
    #                                            sad_codebook, hist_file_extension='_histogram_sad.npy')
    # map_kps_to_sad_codebook_small = gen_histograms(training_descriptors, test_descriptors,
    #                                            training_keypoints, test_keypoints,
    #                                            sad_codebook_small, hist_file_extension='_histogram_sad_small.npy')

