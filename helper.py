import numpy as np
import fnmatch, os, collections
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

CODEBOOK_FILE_TRAIN = f'{DATASET_DIR}/Training/codebook.npy'
CODEBOOK_FILE_TEST = f'{DATASET_DIR}/Test/codebook.npy'

################################################################################
# Read binary files
################################################################################

def read_codebook() -> List[Dict[str, List[int]]]:
    load_training_codebook = np.load(CODEBOOK_FILE_TRAIN, allow_pickle=True)
    load_test_codebook = np.load(CODEBOOK_FILE_TEST, allow_pickle=True)
    return load_training_codebook.tolist(), load_test_codebook.tolist()

def get_histogram_paths():
    training_histogram_paths = collections.defaultdict(list)
    test_histogram_paths = collections.defaultdict(list)

    for class_name in CLASSES:
        training_directory, test_directory = f'{TRAINING_PATH}/{class_name}', f'{TEST_PATH}/{class_name}'

        for file in os.listdir(training_directory):
            if fnmatch.fnmatch(file, '*histogram.npy'):
                training_histogram_paths[(class_name, training_directory)].append(file)

        for file in os.listdir(test_directory):
            if fnmatch.fnmatch(file, '*histogram.npy'):
                test_histogram_paths[(class_name, test_directory)].append(file)

    return training_histogram_paths, test_histogram_paths

def read_histograms(histograms_file_paths):
    histogram_values = collections.defaultdict(list)
    for path_key in histograms_file_paths:
        for file in histograms_file_paths[path_key]:
            load_histograms_values = np.load(f'{path_key[1]}/{file}', allow_pickle=True)
            histogram_values[path_key].append(load_histograms_values)
    return histogram_values