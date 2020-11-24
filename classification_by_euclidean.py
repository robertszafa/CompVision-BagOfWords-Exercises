from typing import List, Dict
import collections, math
import numpy as np
import generate_codebook as gc
import helper as hp

################################################################################
# Step 4. Classification
################################################################################

# Refactor intersection into one file once classification is done

def apply_nearest_neighbour(test_hist: list, training_hist_by_classes: dict) -> Dict[str, int]:
    # Computes the nearest neighbour of the different class to the test histogram
    # limit is used to limit the amount of training histogram to use for each class
    result = collections.defaultdict(int)

    for class_type, class_histograms in training_hist_by_classes.items():
        for train_hist in class_histograms:
            result[class_type] += hp.euclidean_distance(test_hist, train_hist)

    # Return class type with smallest distance.
    return min(result, key=result.get)

def label_all_test_images():
    # Get the paths of all histogram files
    test_path, training_path = hp.get_histogram_paths()

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
            label = apply_nearest_neighbour(hist, all_training_hist)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')

    return images_and_labels


if __name__ == "__main__":
    result = label_all_test_images()
    print(result)
    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])


