from typing import List, Dict
import collections, math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import generate_codebook as gc
import helper as hp

################################################################################
# Result visualisations
################################################################################

def display_image_with_label(label, image_path):
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.title(label)
    plt.show()

def display_multiple_image_with_labels(class_type, images_and_labels):
    rows = 2 
    cols = len(images_and_labels) // 2
    figure, ax = plt.subplots(nrows=rows, ncols=cols)

    for i, image_label_object in enumerate(images_and_labels):
        img = mpimg.imread(f'{class_type[1]}/{image_label_object[1]}')
        label = f'{image_label_object[0]} \n {image_label_object[1]}'
        ax.ravel()[i].imshow(img)
        ax.ravel()[i].set_title(label)
        ax.ravel()[i].set_axis_off()

    plt.tight_layout()
    plt.show()

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

    # Get all the test images directory
    test_images_dir = hp.get_image_paths("jpg")

    # display images with classified label
    images_and_labels = collections.defaultdict(list)

    for class_type in all_test_hist:
        correct_label = 0
        amount = 0
        for hist in all_test_hist[class_type]:
            label = label_classification(hist, all_training_hist, limit)
            print(class_type)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')
    
    return images_and_labels


if __name__ == "__main__":
    # Classification
    s = hp.get_image_paths("jpg")
    result = label_all_test_images(50)
    for key in result:
         display_multiple_image_with_labels(key, result[key])
