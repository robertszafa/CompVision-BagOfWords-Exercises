"""
Commonly used constants and methods for CW1-COMP338.

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import fnmatch, os, collections, re, math
from typing import List, Dict, Set

################################################################################
# Constants
################################################################################

DATASET_DIR = 'COMP338_Assignment1_Dataset'
CLASSES = [
    'airplanes',
    'cars',
    'dog',
    'faces',
    'keyboard',
]
TRAINING_PATH  = f'{DATASET_DIR}/Training'
TEST_PATH = f'{DATASET_DIR}/Test'

CODEBOOK_FILE = f'{DATASET_DIR}/Training/codebook.npy'
CODEBOOK_SMALL_FILE = f'{DATASET_DIR}/Training/codebook_small.npy'
CODEBOOK_EUCLIDEAN_FILE = f'{DATASET_DIR}/Training/codebook_euclidean.npy'
CODEBOOK_EUCLIDEAN_SMALL_FILE = f'{DATASET_DIR}/Training/codebook_euclidean_small.npy'

MAP_KPS_TO_CODEBOOK_FILE = f'{DATASET_DIR}/map_kps_to_codebook.npy'
MAP_KPS_TO_CODEBOOK_SMALL_FILE = f'{DATASET_DIR}/map_kps_to_codebook_small.npy'
MAP_KPS_TO_CODEBOOK_EUCLIDEAN_FILE = f'{DATASET_DIR}/map_kps_to_codebook_euclidean.npy'
MAP_KPS_TO_CODEBOOK_EUCLIDEAN_SMALL_FILE = f'{DATASET_DIR}/map_kps_to_codebook_euclidean_small.npy'

HISTOGRAM_FILE_EXT = "_histogram.npy"
HISTOGRAM_SMALL_FILE_EXT = "_histogram_small.npy"
HISTOGRAM_EUCLIDEAN_FILE_EXT = "_histogram_euclidean.npy"
HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT = "_histogram_euclidean_small.npy"

DEFAULT_IMAGE_FORMAT = "jpg"
LONG_LOCOMOTIVE = "========================================="

################################################################################
# Common math functions
################################################################################

def euclidean_distance(vec1, vec2):
    total = 0
    for i in range(len(vec1)):
        total += pow(vec1[i] - vec2[i], 2)

    return math.sqrt(total)

def sad(vec1, vec2):
    total = 0
    for i in range(len(vec1)):
        total += abs(vec1[i] - vec2[i])

    return total

def mean(vectors):
    """
    Given a list of vectors, return the average vector.
    """
    return np.sum(vectors, 0) / len(vectors)

def k_NN(candidate: list, neighbours_by_class: dict, k=1, dist_func=euclidean_distance):
    """
    Return the class of the k-Nearest Neighbour.
    """
    # [class_of_img_hist: distance] pairs
    results = []

    for class_type, class_histograms in neighbours_by_class.items():
        for train_hist in class_histograms:
            dist = dist_func(candidate, train_hist)
            results.append([class_type, dist])

    # Sort [class_of_img_hist: distance] pairs based on distance
    results.sort(key=lambda res : res[1])

    class_count = {c: 0 for c in neighbours_by_class.keys()}
    for i in range(k):
        class_count[results[i][0]] += 1

    return max(class_count, key=class_count.get)

def get_idx_of_1_NN(candidate, neighbours, dist_func=euclidean_distance):
    min_idx = 0
    min_dist = dist_func(candidate, neighbours[min_idx])

    for i in range(1, len(neighbours)):
        this_dist = dist_func(candidate, neighbours[i])

        if this_dist < min_dist:
            min_dist = this_dist
            min_idx = i

    return min_idx


################################################################################
# Get directory or file paths
################################################################################
def get_image_paths(image_format=DEFAULT_IMAGE_FORMAT, path=TEST_PATH):
    image_paths = collections.defaultdict(list)

    for class_name in CLASSES:
        directory = f'{path}/{class_name}'
        for file in os.listdir(directory):
            if fnmatch.fnmatch(file, f'*.{image_format}'):
                image_paths[(class_name, directory)].append(file)
    return image_paths

def get_histogram_paths(fname_ext=HISTOGRAM_FILE_EXT):
    training_histogram_paths = collections.defaultdict(list)
    test_histogram_paths = collections.defaultdict(list)

    match_fnames = f'*{fname_ext}'

    for class_name in CLASSES:
        training_directory, test_directory = f'{TRAINING_PATH}/{class_name}', f'{TEST_PATH}/{class_name}'

        for file in os.listdir(training_directory):
            if fnmatch.fnmatch(file, match_fnames):
                training_histogram_paths[(class_name, training_directory)].append(file)

        for file in os.listdir(test_directory):
            if fnmatch.fnmatch(file, match_fnames):
                test_histogram_paths[(class_name, test_directory)].append(file)

    return test_histogram_paths, training_histogram_paths

################################################################################
# Read/write binary files
################################################################################
def load_pickled_list(fname) -> List:
    l = []
    with open(fname, 'rb') as f:
        l = np.load(f, allow_pickle=True)
    return l.tolist()

def load_all_histograms(histograms_file_paths):
    histogram_values = collections.defaultdict(list)

    for path_key in histograms_file_paths:
        for file in histograms_file_paths[path_key]:
            load_histograms_values = np.load(f'{path_key[1]}/{file}', allow_pickle=True)
            histogram_values[path_key].append(load_histograms_values.tolist())

    return histogram_values

def initialise_histograms(hist_ext):
    # Get the paths of all histograms
    test_path, training_path = get_histogram_paths(hist_ext)

    # Load all histogram binary file name
    all_test_hist = load_all_histograms(test_path)
    all_training_hist = load_all_histograms(training_path)

    return all_test_hist, all_training_hist

def load_images_in_directory(path) -> Dict[str, List]:
    images = {}
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename] = img

    return images

def load_np_pickles_in_directory(path, regex=r'.*.(npy|npc)'):
    """
    Given a {path} to a directory load all numpy pickle files in that directory that match {regex}.
    Return a dictionary, {fname: np.load(fname)}, where fname includes only the part before '.' and '_'.
    """
    result = {}
    for filename in os.listdir(path):
        if re.match(regex, filename):
            # Get rid of file extensions and (keypoints|descriptors) annotations.
            key = filename.split('.')[0].split('_')[0]
            result[key] = np.load(path + filename, allow_pickle=True)

    return result

def load_descriptors(test_or_train, merge_in_class=False):
    """
    Read the descriptors from the {test_or_train} dataset.
    Return a dictionary with the class names as keys.
    If {merge_in_class} is True, then a single class will have a list of all descriptors as the value.
    Otherwise, it will have a list of dictionaries as values, where the dictionaries have
    the individual img filename as key and their list of descriptors as values.

    if not merge_in_class:
    descriptors = {
        cars: {
            img0: [descriptor1, descriptor2, ....],
            img1: [descriptor1, descriptor2, ....],
            ...
        },
        airplanes: {...}
        ...
    }
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

def save_to_pickle(pickle_fname, data):
    with open(pickle_fname, 'wb') as f:
        np.save(f, data)


################################################################################
# Result visualisations
################################################################################
def display_image_with_label(label, image_path):
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.title(label)
    plt.show()

def display_multiple_image_with_labels(class_type, images_and_labels):
    rows = 2
    cols = len(images_and_labels) // 2

    mpl.rcParams['toolbar'] = 'None'
    figure, ax = plt.subplots(nrows=rows, ncols=cols, num=None, figsize=(10,6), dpi=80,
                              facecolor='w', edgecolor='k')
    figure.canvas.set_window_title(f'Predicting {class_type[0].capitalize()} Images')

    for i, image_label_object in enumerate(images_and_labels):
        img = mpimg.imread(f'{class_type[1]}/{image_label_object[1]}')
        label = f'Predicted: {image_label_object[0]} \n Image File: {image_label_object[1]}'
        ax.ravel()[i].imshow(img)
        ax.ravel()[i].set_title(label)
        ax.ravel()[i].set_axis_off()

    plt.tight_layout()

    # SHow and cloase after a button is pressed.
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()
