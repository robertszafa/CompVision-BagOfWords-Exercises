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

def apply_nearest_neighbour(test_hist, training_hist_by_classes, limit=None) -> Dict[str, int]:
    # Computes the nearest neighbour of the different class to the test histogram
    # limit is used to limit the amount of training histogram to use for each class
    result = collections.defaultdict(int)

    for class_type in training_hist_by_classes:
        n = limit if limit else len(training_hist_by_classes[class_type])
        for hist in range(n):
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += euclidean_distance(test_hist, train_hist)

    return result


def label_classification(test_hist: dict, train_hist: dict, limit=None):
    # Classify one test histogram given multiple training histograms of multiple classes
    result = apply_nearest_neighbour(test_hist, train_hist, limit)
    label = None

    for key in result:
        if not label:
            label = (key[0], result[key])
        elif result[key] < label[1]:
            label = (key[0], result[key])
        out = f'Class Type: {key[0]}- Values: {result[key]}'
        
    return label

def label_all_test_images(limit=50):
    # Get the paths of all histogram files
    training_path, test_path = hp.get_histogram_paths()

    # Get all test histogram keys
    all_test_keys = hp.test_histogram_keys()

    # Get all training histogram keys
    all_train_keys = hp.training_histogram_keys()

    # Load all histogram binary file name
    all_test_hist = hp.load_all_histograms(test_path)
    all_training_hist = hp.load_all_histograms(training_path)

    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            result = label_classification(hist, all_training_hist, limit)
            amount += 1
            if result[0] == class_type[0]:
                correct_label += 1
        percentage_correct = (correct_label / amount) * 100
        print(f'{class_type[0]} correct label is {percentage_correct}%')

            

if __name__ == "__main__":
    label_all_test_images(50)
    
    
    # Classification
    # result = label_classification(single_car, all_training_hist, limit=50)
    # print(result)

# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html

################################################################################
# Step 6. Intersection between two histograms
################################################################################