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
    # Computes the nearest neighbour of the different class to the test histogram
    result = collections.defaultdict(int)

    for class_type in training_hist_by_classes:
        n = limit if limit else len(training_hist_by_classes[class_type])
        for hist in range(n):
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += euclidean_distance(test_hist, train_hist)

    return result


def label_classification(test_hist, train_hist, limit=None):
    # Takes one test histogram of an image and multiple histograms of different classes
    # Compute the nearest neighbour and classify or label the test as one of the class
    result = apply_nearest_neighbour(test_hist, train_hist, limit)

    label = None

    for key in result:
        if not label:
            label = (key[0], result[key])
        elif result[key] < label[1]:
            label = (key[0], result[key])
        out = f'Class Type: {key[0]}- Values: {result[key]}'
        print(out)

    return label


if __name__ == "__main__":
    training_path, test_path = hp.get_histogram_paths()

    # Test Histogram
    all_test_keys = hp.test_histogram_keys()

    # Train Histogram
    all_train_keys = hp.training_histogram_keys()

    # Load all trainign histogram by multiple classes
    all_test_hist = hp.read_all_histograms(test_path)
    all_training_hist = hp.read_all_histograms(training_path)

    print(all_test_keys)

    car_test_key = all_test_keys[1]
    single_car = all_test_hist[car_test_key][3] 

    dog_test_key = all_test_keys[2]
    single_dog = all_test_hist[dog_test_key][5]

    face_test_key = all_test_keys[3]
    single_face = all_test_hist[face_test_key][2]

    keyboard_test_key = all_test_keys[4]
    single_keyboard = all_test_hist[keyboard_test_key][3]
    
    # Classification
    result = label_classification(single_keyboard, all_training_hist, limit=50)
    print(result)

# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html

################################################################################
# Step 6. Intersection between two histograms
################################################################################