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

def apply_nearest_neighbour(training_histogram, test_histogram):
    result = collections.defaultdict(dict)
    for class_type in gc.CLASSES:
        result[class_type] = euclidean_distance(training_histogram[class_type], test_histogram[class_type])
    return result

def label_classification(training_histogram, test_histogram):
    return


if __name__ == "__main__":
    training_path, test_path = hp.get_histogram_paths()

    # Test Histogram
    all_test_keys = hp.test_histogram_keys()
    test_airplanes_key = all_test_keys[0]

    # Train Histogram
    all_train_keys = hp.training_histogram_keys()
    train_airplane_key = all_train_keys[0]

    # Load airplane test histogram and load all class types for training histogram
    airplanes_test_hist = hp.read_single_histograms(test_path, test_airplanes_key)
    all_training_hist = [hp.read_single_histograms(training_path,all_train_keys[i]) for i in range(len(all_train_keys))]

    print(len(all_training_hist[0][train_airplane_key])) # 74 airplane training hist

    # Match single test airplane image vs a single of all classes from training histogram
    test_airplane = airplanes_test_hist[test_airplanes_key][0]

    print(airplanes_test_hist[test_airplanes_key][0])
    print(airplanes_test_hist[test_airplanes_key][1])

    # for i in range(len(all_training_hist)):
    #     a = all_training_hist[i][all_train_keys[i]][0]
    #     print(euclidean_distance(test_airplane, a))

# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html

################################################################################
# Step 6. Intersection between two histograms
################################################################################