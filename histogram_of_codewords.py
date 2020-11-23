import cv2
import numpy as np
import time
import sys
import re

from helper import load_descriptors, load_keypoints, load_pickled_list, save_to_pickle
from helper import DATASET_DIR, CLASSES, CODEBOOK_FILE_TRAIN, SMALL_CODEBOOK_FILE_TRAIN, MAP_KPS_TO_CODEWORD_FILE, MAP_KPS_TO_SMALL_CODEWORD_FILE

###########################################################################
# Step 3. Image representation with a histogram of codewords
################################################################################
## Step 3.1
def euclidean_distance(vec1, vec2):
    diff = vec1 - vec2

    dist = 0
    for p in diff:
        # Euclidean distance in a 2D plane is: np.sqrt(p.x ** 2 + p.y**2). Our p id 1D,
        # so we can just take the absolute value.
        dist += abs(p)

    return dist

def find_nearest_cluster_idx(descriptor, clusters):
    min_idx = 0
    min_dist = euclidean_distance(descriptor, clusters[min_idx])

    for i in range(1, len(clusters)):
        if euclidean_distance(descriptor, clusters[i]) < min_dist:
            min_dist = euclidean_distance(descriptor, clusters[i])
            min_idx = i

    return min_idx

## Step 3.2
def gen_histogram_of_codewords(img_descriptors, codebook):
    """
    Generate a histogram of codewords for a single image, which is represented by a list of features.
    """
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
                title = title=f'{img_fname.split(DATASET_DIR)[1]} --> codeword_{word_idx}'
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


if __name__ == "__main__":
    start_time = time.time()

    codebook = load_pickled_list(CODEBOOK_FILE_TRAIN)
    # codebook = load_pickled_list(SMALL_CODEBOOK_FILE_TRAIN)

    training_descriptors = load_descriptors(test_or_train='Training', merge_in_class=False)
    test_descriptors = load_descriptors(test_or_train='Test', merge_in_class=False)

    # Keep track of indexes of keypoints which mapped to the same codeword. One dictionary of
    # {img_fname: [keypoints]} pairs per codeword.
    map_kps_to_codewords = [dict() for _ in range(len(codebook))]
    # We're not interested in small keypoints for the visualisation.
    KEYPOINT_DIAMETER_THRESHOLD = 30

    # The same layour for the keypoints.
    # Note that keypoint i of a given image corresponds to descriptor i of that image.
    training_keypoints = load_keypoints(test_or_train='Training', merge_in_class=False)
    test_keypoints = load_keypoints(test_or_train='Test', merge_in_class=False)

    training_histogram_of_codewords = {c_name: dict() for c_name in CLASSES}
    for img_class, descriptors_files in training_descriptors.items():
        for img_id, img_descriptors in descriptors_files.items():
            img_histogram, descriptor_to_codeword_map = gen_histogram_of_codewords(img_descriptors, codebook)

            nor_img_histogram = normalise_histogram(img_histogram)
            training_histogram_of_codewords[img_class][img_id] = nor_img_histogram

            # Use the fact that there is a 1:1 mapping between descriptor and kypoint idxs.
            # Use full img path, instead of id, for easier visualisation.
            img_fname = f'{DATASET_DIR}/Training/{img_class}/{img_id}.jpg'
            for word_idx, keypoint_idxs_list in enumerate(descriptor_to_codeword_map):
                # Get rid of small keypoints.
                filtered_keypoints = []
                for kp_idx in keypoint_idxs_list:
                    # We have saved the keypoint as [(kp_x, (kp_y), kp_diameter]
                    kp = training_keypoints[img_class][img_id][kp_idx]
                    if kp[1] > KEYPOINT_DIAMETER_THRESHOLD:
                        filtered_keypoints.append(kp)

                map_kps_to_codewords[word_idx][img_fname] = filtered_keypoints

            # Save each image histogram to a seperate file
            hist_fname = f'{DATASET_DIR}/Training/{img_class}/{img_id}_histogram.npy'
            # hist_fname = f'{DATASET_DIR}/Training/{img_class}/{img_id}_histogram_small.npy'
            save_to_pickle(hist_fname, nor_img_histogram)

    # Same for Test images.
    test_histogram_of_codewords = {c_name: dict() for c_name in CLASSES}
    for img_class, images in test_descriptors.items():
        for img_id, img_descriptors in images.items():
            img_histogram, descriptor_to_codeword_map = gen_histogram_of_codewords(img_descriptors, codebook)

            nor_img_histogram = normalise_histogram(img_histogram)
            test_histogram_of_codewords[img_class][img_id] = nor_img_histogram

            # Use the fact that there is a 1:1 mapping between descriptor and kypoint idxs.
            # Use full img path, instead of id, for easier visualisation.
            img_fname = f'{DATASET_DIR}/Test/{img_class}/{img_id}.jpg'
            for word_idx, keypoint_idxs_list in enumerate(descriptor_to_codeword_map):
                # Get rid of small keypoints.
                filtered_keypoints = []
                for kp_idx in keypoint_idxs_list:
                    # We have saved the keypoint as [(kp_x, (kp_y), kp_diameter]
                    kp = test_keypoints[img_class][img_id][kp_idx]
                    if kp[1] > KEYPOINT_DIAMETER_THRESHOLD:
                        filtered_keypoints.append(kp)

                map_kps_to_codewords[word_idx][img_fname] = filtered_keypoints

            # Save each image histogram to a seperate file
            hist_fname = f'{DATASET_DIR}/Test/{img_class}/{img_id}_histogram.npy'
            # hist_fname = f'{DATASET_DIR}/Test/{img_class}/{img_id}_histogram_small.npy'
            save_to_pickle(hist_fname, nor_img_histogram)


    # map_kps_to_codewords = load_pickled_list(MAP_KPS_TO_CODEWORD_FILE)
    save_to_pickle(MAP_KPS_TO_CODEWORD_FILE, map_kps_to_codewords)

    # map_kps_to_codewords = load_pickled_list(MAP_KPS_TO_SMALL_CODEWORD_FILE)
    # save_to_pickle(MAP_KPS_TO_SMALL_CODEWORD_FILE, map_kps_to_codewords)

    # Put most matched codewords and the corresponding keypoints to the front.
    # Note that we don't care towhich specific codeword given keypoints match, we just care about
    # the fact that they match to the same one.
    num_of_kps = lambda d : np.sum([len(v) for v in d.values()])
    map_kps_to_codewords.sort(key=num_of_kps, reverse=True)

    # Step 3.3 Visualize some image patches that are assigned to the same codeword.
    visualize_similar_patches(map_kps_to_codewords)

    print(f'Finished program in {(time.time() - start_time)/60} minutes.')
