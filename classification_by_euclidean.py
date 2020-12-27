"""
CW1-COMP338 - Step 4 & 5 Classification and evaluation

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import argparse
from typing import List, Dict
import collections, math
import numpy as np
import helper as hp

################################################################################
# Step 4. Classification
################################################################################
def label_all_test_images(hist_ext=hp.HISTOGRAM_FILE_EXT, k=1):
    all_test_hist, all_training_hist = hp.initialise_histograms(hist_ext)               # Get all the test and training histograms
    test_images_dir = hp.get_image_paths(hp.DEFAULT_IMAGE_FORMAT, hp.TEST_PATH)         # Get all the test images directory
    images_and_labels = collections.defaultdict(list)                                   # display images with classified label

    num_words = len(list(all_test_hist.values())[0][0])
    dictionary_dist_func = 'euclidean' if 'euclidean' in hist_ext else 'Sum Of Absolute Difference'
    print(f'---> Using histograms generated from a {num_words}-word dictionary')
    print(f'---> Dictionary was clusterd using {dictionary_dist_func} distance function')
    print(f'---> Using {k}-Nearest Neighbour to classify')

    total_error = 0
    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            label = hp.k_NN(hist, all_training_hist, dist_func=hp.euclidean_distance, k=k)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1

        percentage_error = int((1 - (correct_label / amount)) * 100)
        total_error += percentage_error
        # hp.print_classification_percentage(class_type[0], percentage_error)
        print(f'{percentage_error}% classification error for the {class_type[0]} class')

    print(f'---> {total_error/len(all_test_hist)}% overall classification error.')
    print(hp.LONG_LOCOMOTIVE)

    return images_and_labels

def label_all_training_images(hist_ext=hp.HISTOGRAM_FILE_EXT, k=5):
    _, all_training_hist = hp.initialise_histograms(hist_ext)
    test_images_dir = hp.get_image_paths(hp.DEFAULT_IMAGE_FORMAT, hp.TRAINING_PATH)
    images_and_labels = collections.defaultdict(list)

    num_words = len(list(all_training_hist.values())[0][0])
    dictionary_dist_func = 'euclidean' if 'euclidean' in hist_ext else 'Sum Of Absolute Difference'
    print(f'---> Using histograms generated from a {num_words}-word dictionary')
    print(f'---> Dictionary was clusterd using {dictionary_dist_func} distance function')

    total_error = 0
    for class_type in all_training_hist:
        correct_label, amount = 0, 0
        for hist in all_training_hist[class_type]:
            # Pass all training historgrams as neighbours to k-NN exclusing the one we want to label.
            pass_to_knn = {c: [h for h in hs if h != hist]
                           for c, hs in all_training_hist.items()}

            label = hp.k_NN(hist, pass_to_knn, dist_func=hp.euclidean_distance, k=k)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))

            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_error = int((1 - (correct_label / amount)) * 100)
        total_error += percentage_error
        print(f'{percentage_error}% classification error for the {class_type[0]} class')

    print(f'---> {total_error/len(all_training_hist)} average preccision.')
    print(hp.LONG_LOCOMOTIVE)

    return images_and_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify the class of images using euclidean distance between histograms.')
    parser.add_argument('-e', help='use codebook generated using euclidean distance', action='store_true')
    parser.add_argument('-s', help='use small codebook', action='store_true')
    parser.add_argument('--training', help='classify training images', action='store_true')
    args = parser.parse_args()

    print("Classification using euclidean distance between histograms... \n" + hp.LONG_LOCOMOTIVE)

    if args.training:
        # We were classifying training images for testing purposes.
        label_func = label_all_training_images
    else:
        label_func = label_all_test_images

    if args.e and args.s:
        result = label_func(hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT, k=23)
    elif args.e:
        result = label_func(hp.HISTOGRAM_EUCLIDEAN_FILE_EXT, k=1)
    elif args.s:
        result = label_func(hp.HISTOGRAM_SMALL_FILE_EXT, k=24)
    else:
        result = label_func(hp.HISTOGRAM_FILE_EXT, k=3)


    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])
