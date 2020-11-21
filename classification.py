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

# {'airplanes': 600.9267842258323, 'cars': 549.5152409169376, 'dog': 607.9662819597811, 'faces': 606.4041556585838, 'keyboard': 597.047736784924})
# Matching between input image and training image is found by calculating the distance between historgram

def label_classification(training_histogram, test_histogram):
    return

if __name__ == "__main__":
    training, test = hp.get_histogram_paths()
    s = hp.read_histograms(test)
    print(s)

# useful sources
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
# https://uk.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html

################################################################################
# Step 6. Intersection between two histograms
################################################################################