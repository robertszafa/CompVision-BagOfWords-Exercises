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

    descriptor_to_codeword_map = [[] for _ in range(len(img_class_codebook))]
    map_descriptor = lambda word_idx, descriptor_idx : \
                      descriptor_to_codeword_map[word_idx].append(descriptor_idx)

    for idx, descriptor in enumerate(img_descriptors):
        closest_cluster_idx = find_nearest_cluster_idx(descriptor, img_class_codebook)
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

def visualize_similar_patches(map_kp_idxs_to_codewords, training_keypoints, imgs_path):
    # map_kp_idxs_to_codewords = {
    #     cars: [
    #             {
    #                 img_id_0: [KeyPoints...],
    #                 img_id_1: [KeyPoints...],
    #                 ...
    #             }, <-- Dict of {img_id: keypoints_list} pairs assigned to codeword 0 from cars class.
    #             ... same wfor rest of codwords in this class.
    #     ]
    for img_class, kp_to_codeword_map in map_kp_idxs_to_codewords.items():
        for word_idx, img_ids_to_kp_dict in enumerate(kp_to_codeword_map):
            for img_id, kp_idxs_list in img_ids_to_kp_dict.items():
                for kp_idx in kp_idxs_list:
                    # We have saved the keypoint as [(kp_x, (kp_y), kp_diameter]
                    kp = training_keypoints[img_class][img_id][kp_idx]

                    img_fname = f'{imgs_path}/{img_class}/{img_id}.jpg'
                    title = title=f'{img_class}/{img_id} --> codeword_{word_idx}'

                    draw_keypoint(img_fname, int(kp[0][0]), int(kp[0][1]), int(kp[1]), title)

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


def load_dict_pickle(fname):
    # Read codebook.
    with open(fname, 'rb') as f:
        tmp_codebook = np.load(f, allow_pickle=True)

    # numpy wraps the python dictionary into an object array.
    # Transform it back into a python dictionary.
    codebook = {}
    for img_class in CLASSES:
        codebook[img_class] = tmp_codebook[()][img_class]

    return codebook



if __name__ == "__main__":
    codebook = load_dict_pickle(CODEBOOK_FILE_TRAIN)

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

    # Keep track of indexes of keypoints which mapped to the same codeword. With this data structure,
    # we'll be able to easily visualise img patches assigned to the same keyword (step 3.3).
    # Note that we're restricting ourselfs to the training images only.
    # map_kp_idxs_to_codewords = {
    #     cars: [
    #             {
    #                 img_id_0: [KeyPoints...],
    #                 img_id_1: [KeyPoints...],
    #                 ...
    #             }, <-- Dict of {img_id: keypoints_list} pairs assigned to codeword 0 from cars class.
    #             ... same wfor rest of codwords in this class.
    #     ],
    #     airplanes: [...]
    #     ...
    # }
    map_kp_idxs_to_codewords = {c_name: [dict() for _ in range(len(codebook[c_name]))] for c_name in CLASSES}
    # We're not interested in small keypoints for the visualisation.
    KEYPOINT_DIAMETER_THRESHOLD = 30

    # The same layour for the keypoints.
    # Note that keypoint i of a given image corresponds to descriptor i of that image.
    training_keypoints = load_keypoints(test_or_train='Training', merge_in_class=False)

    training_histogram_of_codewords = {c_name: dict() for c_name in CLASSES}
    for img_class, descriptors_files in training_descriptors.items():
        for img_id, img_descriptors in descriptors_files.items():
            img_histogram, descriptor_to_codeword_map = \
                gen_histogram_of_codewords(img_descriptors, codebook[img_class])

            nor_img_histogram = normalise_histogram(img_histogram, codebook[img_class], l1_norm)
            training_histogram_of_codewords[img_class][img_id] = nor_img_histogram

            # Use the fact that there is a 1:1 mapping between descriptor and kypoint idxs.
            for word_idx, keypoint_idxs_list in enumerate(descriptor_to_codeword_map):
                filtered_keypoint_idxs_list = [kp_idx for kp_idx in keypoint_idxs_list
                    if training_keypoints[img_class][img_id][kp_idx][1] > KEYPOINT_DIAMETER_THRESHOLD]
                map_kp_idxs_to_codewords[img_class][word_idx][img_id] = filtered_keypoint_idxs_list

            # Save each image histogram to a seperate file
            fname = f'{img_id}_histogram.npy'
            with open(f'{DATASET_DIR}/Training/{img_class}/{fname}', 'wb') as f:
                np.save(f, nor_img_histogram)

    # Same for Test images.
    test_histogram_of_codewords = {c_name: dict() for c_name in CLASSES}
    for img_class, images in test_descriptors.items():
        for img_id, img_descriptors in images.items():
            img_histogram, _ = gen_histogram_of_codewords(img_descriptors, codebook[img_class])

            nor_img_histogram = normalise_histogram(img_histogram, codebook[img_class], l1_norm)
            test_histogram_of_codewords[img_class][img_id] = nor_img_histogram

            # Save each image histogram to a seperate file
            fname = f'{img_id}_histogram.npy'
            with open(f'{DATASET_DIR}/Test/{img_class}/{fname}', 'wb') as f:
                np.save(f, nor_img_histogram)


    with open('map_kp_idxs_to_codewords.npy', 'wb') as f:
        np.save(f, map_kp_idxs_to_codewords)
    # map_kp_idxs_to_codewords = load_dict_pickle('map_kp_idxs_to_codewords.npy')

    # Put most matched codewords and the corresponding keypoint_idxs to the front.
    for img_class in map_kp_idxs_to_codewords.keys():
        num_of_kps = lambda d : np.sum([len(v) for v in d.values()])
        map_kp_idxs_to_codewords[img_class].sort(key=num_of_kps, reverse=True)


    # Step 3.3 Visualize some image patches that are assigned to the same codeword.
    visualize_similar_patches(map_kp_idxs_to_codewords, training_keypoints,
                              imgs_path=f'{DATASET_DIR}/Training/')
