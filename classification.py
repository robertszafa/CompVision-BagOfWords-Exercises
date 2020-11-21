from typing import List, Dict
import collections
import math
import generate_codebook as gc
import helper as hp

L1_NORM = collections.namedtuple("L1_norm", "value")

################################################################################
# Step 4. Classification
################################################################################

def euclidean_distance(class_type, train_hist, test_hist):
    total = 0
    for i in range(len(train_hist)):
        total += pow(train_hist[i].value - test_hist[i].value, 2)
    return math.sqrt(total)

def apply_nearest_neighbour(training_histogram, test_histogram):
    result = collections.defaultdict(dict)
    for class_type in gc.CLASSES:
        result[class_type] = euclidean_distance(class_type, training_histogram[class_type], test_histogram[class_type])
    return result

def label_classification(training_histogram, test_histogram):
    return


def testing(training, test, train_airplanes_key, test_airplanes_key):
    test_histograms = hp.read_all_histograms(test)
    training_histograms = hp.read_all_histograms(training)

    test_a1 = test_histograms[test_airplanes_key][0] # Takes too long to load
    train_a1 = training_histograms[train_airplanes_key][0]
    train_a2 = training_histograms[train_airplanes_key][1]

if __name__ == "__main__":
    training_path, test_path = hp.get_histogram_paths()
    train_airplanes_key = (hp.CLASSES[0], f'{hp.TRAINING_PATH}/{hp.CLASSES[0]}')
    test_airplanes_key = (hp.CLASSES[0], f'{hp.TEST_PATH}/{hp.CLASSES[0]}')

    # Testing test airplanes vs training airplanes histogram of codewords
    a_test = hp.read_single_histograms(test_path, test_airplanes_key)
    a_training = hp.read_single_histograms(training_path,train_airplanes_key)

    a_test_size = len(a_test[test_airplanes_key])


# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html

################################################################################
# Step 6. Intersection between two histograms
################################################################################