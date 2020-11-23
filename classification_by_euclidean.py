from typing import List, Dict
import collections, math
import numpy as np
import generate_codebook as gc
import helper as hp

################################################################################
# Step 4. Classification
################################################################################

# Refactor intersection into one file once classification is done

def euclidean_distance(test_hist, train_hist) -> int:
    total = 0
    for i, codeword_freq in enumerate(test_hist):
        total += pow(test_hist[i] - train_hist[i], 2)
    return math.sqrt(total)

def apply_nearest_neighbour(test_hist, training_hist_by_classes, limit=None) -> Dict[str, int]:
    # Computes the nearest neighbour of the different class to the test histogram
    # limit is used to limit the amount of training histogram to use for each class
    result = collections.defaultdict(int)
    count = 0

    for class_type in training_hist_by_classes:
        n = limit if limit else len(training_hist_by_classes[class_type])
        for hist in range(n):
            count += 1
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += euclidean_distance(test_hist, train_hist)

    return result

def label_classification(test_hist: list, train_hist: dict, limit=None):
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
            label = label_classification(hist, all_training_hist, limit)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')
    
    return images_and_labels


if __name__ == "__main__":
    result = label_all_test_images(limit=50)
    print(result)
    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])


