from typing import Dict, List
import helper as hp
import collections

################################################################################
# Step 6. Intersection between two histograms
################################################################################

# Same number of bin (i.e. codewords)
def intersection(test_hist, train_hist):
    total = 0
    for i, codeword_freq in enumerate(test_hist):
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
            label = label_histogram_by_intersection(hist, all_training_hist)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            amount += 1
            if label[0] == class_type[0]:
                correct_label += 1
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
