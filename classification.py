from typing import List, Dict
from random import randint
import collections
import math
import generate_codebook as gc

L1_NORM = collections.namedtuple("L1_norm", "value")

################################################################################
# Step 4. Classification
################################################################################

def euclideanDistance(class_type, train_hist, test_hist):
    total = 0
    for i in range(len(train_hist)):
        total += pow(train_hist[i].value - test_hist[i].value, 2)
    return math.sqrt(total)

def apply_nearest_neighbour(training_histogram, test_histogram):
    result = []
    for class_type in gc.CLASSES:
        result.append(euclideanDistance(class_type, training_histogram[class_type], test_histogram[class_type]))
    return result

def label_classification(training_histogram, test_histogram):
    # Assign label/class to an image
    return

def gen_test_histogram():
    histogram = {}
    for class_type in gc.CLASSES:
        for _ in range(0, 128):
            random = randint(0, 128)
            if class_type in histogram:
                histogram[class_type].append(L1_NORM(random)) 
            else:
                histogram[class_type] = [L1_NORM(random)]
    return histogram

if __name__ == "__main__":
    training_test = gen_test_histogram()
    test_test = gen_test_histogram()
    apply_nn = apply_nearest_neighbour(training_test, test_test)
    print(apply_nn)
