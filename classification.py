from typing import List, Dict
import collections, math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
        
    return label

def label_all_test_images(limit=None):
    # Get the paths of all histogram files
    training_path, test_path = hp.get_histogram_paths()

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
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')

################################################################################
# Step 6. Intersection between two histograms
################################################################################

# Same number of bin (i.e. codewords)
def intersection(test_hist, train_hist):
    total = 0
    for codeword_freq in train_hist:
        total += min(test_hist[codeword_freq], train_hist[codeword_freq])
    return total

def apply_intersection(test_hist, training_hist_by_classes, limit=None) -> Dict[str, int]:
    result = collections.defaultdict(int)

    for class_type in training_hist_by_classes:
        n = limit if limit else len(training_hist_by_classes[class_type])
        for hist in range(n):
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += intersection(test_hist, train_hist)
    return result

def label_histogram_by_intersection(test_hist, train_hist, limit=None):
    # Classify one test histogram given multiple training histograms of multiple classes
    result = apply_intersection(test_hist, train_hist, limit)
    label = None

    for key in result:
        if not label:
            label = (key[0], result[key])
        elif result[key] > label[1]:
            label = (key[0], result[key])
    return label

def label_by_intersection(limit=None):
    # Get the paths of all histogram files
    training_path, test_path = hp.get_histogram_paths()

    # Load all histogram binary file name
    all_test_hist = hp.load_all_histograms(test_path)
    all_training_hist = hp.load_all_histograms(training_path)

    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            result = label_histogram_by_intersection(hist, all_training_hist, limit)
            amount += 1
            if result[0] == class_type[0]:
                correct_label += 1
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')

def display_image_with_label(label, image_path):
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.title(label)
    plt.show()

if __name__ == "__main__":
    # Classification
    s = hp.get_image_paths("jpg")
    img = s['airplanes'][0]
    display_image_with_label("airplanes", img)

# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html
