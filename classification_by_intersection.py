"""
CW1-COMP338 - Step 6 Classification by histogram intersection

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import argparse
from typing import Dict, List
import helper as hp
import collections

################################################################################
# Step 6. Intersection between two histograms
################################################################################

# Same number of bin (i.e. codewords)
def intersection(test_hist, train_hist):
    total = 0
    for i in range(len(test_hist)):
        total += min(test_hist[i], train_hist[i])
    return total

def apply_intersection(test_hist, training_hist_by_classes) -> Dict[str, int]:
    result = collections.defaultdict(int)

    for class_type in training_hist_by_classes:
        for hist in range(len(training_hist_by_classes[class_type])):
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += intersection(test_hist, train_hist)
    return result

def label_histogram_by_intersection(test_hist, train_hist):
    # Classify one test histogram given multiple training histograms of multiple classes
    result = apply_intersection(test_hist, train_hist)
    label = None

    for key in result:
        if not label:
            label = (key[0], result[key])
        elif result[key] > label[1]:
            label = (key[0], result[key])
    return label

def label_all_test_images(hist_ext=hp.HISTOGRAM_FILE_EXT):
    all_test_hist, all_training_hist = hp.initialise_histograms(hist_ext)               # Get all the test and training histograms
    test_images_dir = hp.get_image_paths(hp.DEFAULT_IMAGE_FORMAT, hp.TEST_PATH)         # Get all the test images directory
    images_and_labels = collections.defaultdict(list)                                   # display images with classified label

    num_words = len(list(all_test_hist.values())[0][0])
    dictionary_dist_func = 'euclidean' if 'euclidean' in hist_ext else 'Sum Of Absolute Difference'
    print(f'---> Using histograms generated from a {num_words}-word dictionary')
    print(f'---> Dictionary was clusterd using {dictionary_dist_func} distance function')

    total_error = 0
    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            label = label_histogram_by_intersection(hist, all_training_hist)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            amount += 1
            if label[0] == class_type[0]:
                correct_label += 1

        percentage_error = int((1 - (correct_label / amount)) * 100)
        total_error += percentage_error
        print(f'{percentage_error}% classification error for the {class_type[0]} class')

    print(f'---> {total_error/len(all_test_hist)}% overall classification error.')
    print(hp.LONG_LOCOMOTIVE)

    return images_and_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify the class of images using histogram intersetion.')
    parser.add_argument('-e', help='use codebook generated using euclidean distance', action='store_true')
    parser.add_argument('-s', help='use small codebook', action='store_true')
    args = parser.parse_args()

    print("Classification using histogram intersetion... \n" + hp.LONG_LOCOMOTIVE)

    if args.e and args.s:
        result = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT)
    elif args.e:
        result = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_FILE_EXT)
    elif args.s:
        result = label_all_test_images(hp.HISTOGRAM_SMALL_FILE_EXT)
    else:
        result = label_all_test_images(hp.HISTOGRAM_FILE_EXT)


    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])
