import cv2
import numpy as np
import time
import sys

from generate_codebook import load_descriptors, load_keypoints

################################################################################
# Constants
################################################################################
DATASET_DIR = 'COMP338_Assignment1_Dataset'
CODEBOOK_FILE_TRAIN = f'{DATASET_DIR}/Training/codebook.npy'
HISTOGRAM_FILE_TRAIN = f'{DATASET_DIR}/Training/histogram.npy'
HISTOGRAM_FILE_TEST = f'{DATASET_DIR}/Test/histogram.npy'
CLASSES = [
    'airplanes',
    'cars',
    'dog',
    'faces',
    'keyboard',
]

################################################################################
# Step 2. Image representation with a histogram of codewords
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
def gen_histogram_of_codewords(img_descriptors, img_class_codebook):
    """
    Generate a histogram of codewords for a single image, which is represented by a list of features.
    """
    # Initially, each image will have a count of 0 for each codeword.
    histogram_of_codewords = [0 for _ in range(len(img_class_codebook))]

    descriptor_to_codeword_count = [[] for _ in range(len(img_class_codebook))]
    map_descriptor = lambda word_idx, descriptor_idx : \
                      descriptor_to_codeword_count[word_idx].append(descriptor_idx)

    for idx, descriptor in enumerate(img_descriptors):
        closest_cluster_idx = find_nearest_cluster_idx(descriptor, img_class_codebook)
        histogram_of_codewords[closest_cluster_idx] += 1

        map_descriptor(closest_cluster_idx, idx)

    # Get the winning list.
    idxs_of_most_similar_descriptors = max(descriptor_to_codeword_count, key=len)

    return histogram_of_codewords, idxs_of_most_similar_descriptors

## Step 3.3
def draw_keypoint(img_fname, kp_x, kp_y, kp_diameter, title=''):
    img = cv2.imread(img_fname)
    radius = kp_diameter//2
    kp = img[kp_x-radius : kp_x+radius][kp_y-radius : kp_y+radius][:]

    cv2.imshow(title, kp)
    cv2.waitKey(0)

def visualize_similar_patches(idxs_of_most_similar_keypoints_dict, training_keypoints, imgs_path):
    for img_class, images in training_descriptors.items():
        for img_fname, idxs_of_most_similar_keypoints in images.items():
            for i, k_idx in enumerate(idxs_of_most_similar_keypoints):
                if img_fname in training_keypoints[img_class]:
                    # We have saved the keypoint as [(kp_x, (kp_y), kp_diameter]
                    kp = training_keypoints[img_class][img_fname][k_idx]

                    img_fname = f'{imgs_path}/{img_class}/{img_fname}'
                    title = title=f'{img_class}/{img_fname} {i}-th similar patch'

                    draw_keypoint(img_fname, kp[0][0], kp[0][1], kp[1], title)
                else:
                    print('not in here')

## Step 3.4
def l1_norm(vec):
    return np.sum(abs(vec))

def normalise_histogram(img_histogram_of_codewords, img_class_codebook, norm_func):
    assert len(img_histogram_of_codewords) == len(img_class_codebook)

    # Normalise the histogram using the passed in norm function.
    # The norm value of the codeword becomes the key and the frequency of the codeword becomes the value.
    normalised_histogram_of_codewords = {}
    for word_idx in range(len(img_histogram_of_codewords)):
        norm = norm_func(img_class_codebook[word_idx])

        if norm in normalised_histogram_of_codewords:
            # Two histograms could have the same norm. Unlikely, but if it does occur,
            # then we'll take the combined frequencies.
            normalised_histogram_of_codewords[norm] += img_histogram_of_codewords[word_idx]
        else:
            normalised_histogram_of_codewords[norm] = img_histogram_of_codewords[word_idx]

    return normalised_histogram_of_codewords


def read_codebook():
    # Read codebook.
    with open(CODEBOOK_FILE_TRAIN, 'rb') as f:
        tmp_codebook = np.load(f, allow_pickle=True)

    # numpy wraps the python dictionary into an object array.
    # Transform it back into a python dictionary.
    codebook = {}
    for img_class in CLASSES:
        codebook[img_class] = tmp_codebook[()][img_class]

    return codebook



if __name__ == "__main__":
    codebook = read_codebook()

    # Get image descriptors, the format will be:
    # training_descriptors = {
    #     cars: {
    #         img0: [descriptor1, descriptor2, ....],
    #         img1: [descriptor1, descriptor2, ....],
    #         ...
    #     },
    #     airplanes: {...}
    #     ...
    # }
    training_descriptors = load_descriptors(test_or_train='Training', merge_in_class=False)
    test_descriptors = load_descriptors(test_or_train='Test', merge_in_class=False)

    # The same layour for the keypoints.
    # Note that keypoint i of a given image corresponds to descriptor i of that image.
    training_keypoints = load_keypoints(test_or_train='Training', merge_in_class=False)
    # Keep track of indexes of keypoints which mapped to the same codeword the most.
    idxs_of_most_similar_keypoints_dict = {c_name: dict() for c_name in CLASSES}

    training_histogram_of_codewords = {c_name: dict() for c_name in CLASSES}
    for img_class, images in training_descriptors.items():
        for img_fname, img_descriptors in images.items():
            img_histogram, idxs_of_most_similar_keypoints = \
                gen_histogram_of_codewords(img_descriptors, codebook[img_class])

            nor_img_histogram = normalise_histogram(img_histogram, codebook[img_class], l1_norm)
            training_histogram_of_codewords[img_class][img_fname] = nor_img_histogram

            idxs_of_most_similar_keypoints_dict[img_class][img_fname] = idxs_of_most_similar_keypoints

    # Same for Test images.
    test_histogram_of_codewords = {c_name: dict() for c_name in CLASSES}
    for img_class, images in test_descriptors.items():
        for img_fname, img_descriptors in images.items():
            img_histogram, _ = gen_histogram_of_codewords(img_descriptors, codebook[img_class])

            nor_img_histogram = normalise_histogram(img_histogram, codebook[img_class], l1_norm)
            test_histogram_of_codewords[img_class][img_fname] = nor_img_histogram


    with open(HISTOGRAM_FILE_TRAIN, 'wb') as f:
        np.save(f, training_histogram_of_codewords)

    with open(HISTOGRAM_FILE_TEST, 'wb') as f:
        np.save(f, test_histogram_of_codewords)

    with open('idxs_of_most_similar_keypoints.npy', 'wb') as f:
        np.save(f, idxs_of_most_similar_keypoints_dict)

    ## Step 3.3 Visualize some image patches that are assigned to the same codeword.
    visualize_similar_patches(idxs_of_most_similar_keypoints_dict,
                              training_keypoints, imgs_path=f'{DATASET_DIR}/Training/')
