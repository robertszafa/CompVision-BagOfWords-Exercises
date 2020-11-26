from typing import List, Dict
import collections, math
import numpy as np

import helper as hp

################################################################################
# Step 4. Classification
################################################################################
def label_all_test_images(hist_ext=hp.HISTOGRAM_FILE_EXT):
    # Get the paths of all histogram files
    test_path, training_path = hp.get_histogram_paths(hist_ext)

    # Load all histogram binary file name
    all_test_hist = hp.load_all_histograms(test_path)
    all_training_hist = hp.load_all_histograms(training_path)

    # Get all the test images directory
    test_images_dir = hp.get_image_paths("jpg")

    # display images with classified label
    images_and_labels = collections.defaultdict(list)

    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            label = hp.k_NN(hist, all_training_hist, dist_func=hp.euclidean_distance, k=1)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')

    return images_and_labels


if __name__ == "__main__":
    result = label_all_test_images(hp.HISTOGRAM_FILE_EXT)
    result_small_codebook = label_all_test_images(hp.HISTOGRAM_SMALL_FILE_EXT)
    result_euclidean_codebook = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_FILE_EXT)
    result_small_euclidean_codebook = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT)

    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])
