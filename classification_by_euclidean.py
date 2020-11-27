from typing import List, Dict
import collections, math
import numpy as np
import helper as hp

def label_all_training_images(hist_ext=hp.HISTOGRAM_FILE_EXT):
    all_test_hist, all_training_hist = hp.initialise_histograms(hist_ext)
    training_image_dir = hp.get_image_paths(hp.DEFAULT_IMAGE_FORMAT, hp.TRAINING_PATH)

    images_and_labels = collections.defaultdict(list)
    print("Starting ")
    for class_type in all_training_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            label = hp.k_NN(hist, all_training_hist, dist_func=hp.euclidean_distance, k=1)
            print("The label is " + label) # Label is returning nothing?
            images_and_labels[class_type].append((label[0], training_image_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        print(images_and_labels)
        # percentage_correct = int((correct_label / amount) * 100)
        # print(f'{class_type[0]} correct label is {percentage_correct}%')

    return images_and_labels


################################################################################
# Step 4. Classification
################################################################################

def label_all_test_images(hist_ext=hp.HISTOGRAM_FILE_EXT):
    all_test_hist, all_training_hist = hp.initialise_histograms(hist_ext)               # Get all the test and training histograms
    test_images_dir = hp.get_image_paths(hp.DEFAULT_IMAGE_FORMAT, hp.TEST_PATH)         # Get all the test images directory
    images_and_labels = collections.defaultdict(list)                                   # display images with classified label

    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            label = hp.k_NN(hist, all_training_hist, dist_func=hp.euclidean_distance, k=1)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        hp.print_classification_percentage(class_type[0], percentage_correct)
    print(hp.LONG_LOCOMOTIVE)
    return images_and_labels

if __name__ == "__main__":
    print("Classification using euclidean distance... \n" + hp.LONG_LOCOMOTIVE)
    result = label_all_test_images(hp.HISTOGRAM_FILE_EXT)
    result_small_codebook = label_all_test_images(hp.HISTOGRAM_SMALL_FILE_EXT)
    result_euclidean_codebook = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_FILE_EXT)
    result_small_euclidean_codebook = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT)

    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])
