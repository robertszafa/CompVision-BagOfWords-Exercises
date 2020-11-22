from typing import List, Dict
import collections
import math
import generate_codebook as gc
import helper as hp

L1_NORM = collections.namedtuple("L1_norm", "value")

################################################################################
# Step 4. Classification
################################################################################

def euclidean_distance(test_hist, train_hist) -> int:
    total = 0
    for codeword_freq in train_hist:
        total += pow(test_hist[codeword_freq] - train_hist[codeword_freq], 2)
    return math.sqrt(total)

def apply_nearest_neighbour(test_hist, training_hist_by_classes, limit=None):
    # Takes a single training

    result = collections.defaultdict(int)

    for class_type in training_hist_by_classes:
        n = limit if limit else len(training_hist_by_classes[class_type])
        for hist in range(n):
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += euclidean_distance(test_hist, train_hist)

    return result


def label_classification(training_histogram, test_histogram):
    return


if __name__ == "__main__":
    training_path, test_path = hp.get_histogram_paths()

    # Test Histogram
    all_test_keys = hp.test_histogram_keys()
    test_airplanes_key = all_test_keys[0]
    test_car_key = all_test_keys[1]

    # Train Histogram
    all_train_keys = hp.training_histogram_keys()

    # Load airplane test histogram and load all class types for training histogram
    airplanes_test_hist = hp.read_single_histograms(test_path, test_airplanes_key)
    all_training_hist = hp.read_all_histograms(training_path)

    # Match single test airplane image vs multiple of all classes from training histogram
    airplane_single_hist_test = airplanes_test_hist[test_airplanes_key][1]
    
    result = apply_nearest_neighbour(airplane_single_hist_test, all_training_hist)
    
    for key in result:
        out = f'Class Type: {key[0]}- Values: {result[key]}'
        print(out)


# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html

################################################################################
# Step 6. Intersection between two histograms
################################################################################