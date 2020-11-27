from typing import List, Dict
import collections, math
import numpy as np
import helper as hp

def label_all_training_images(hist_ext=hp.HISTOGRAM_FILE_EXT, k=5):
    _, all_training_hist = hp.initialise_histograms(hist_ext)               # Get all the test and training histograms

    num_words = len(list(all_training_hist.values())[0][0])
    dictionary_dist_func = 'euclidean' if 'euclidean' in hist_ext else 'Sum Of Absolute Difference'
    print(f'---> Using histograms generated from a {num_words}-word dictionary')
    print(f'---> Dictionary was clusterd using {dictionary_dist_func} distance function')

    total_preccision = 0
    for class_type in all_training_hist:
        correct_label, amount = 0, 0
        for hist in all_training_hist[class_type]:
            # Pass all training historgrams as neighbours to k-NN exclusing the one we want to label.
            pass_to_knn = {c: [h for h in hs if h != hist]
                           for c, hs in all_training_hist.items()}

            label = hp.k_NN(hist, pass_to_knn, dist_func=hp.euclidean_distance, k=k)

            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        total_preccision += percentage_correct
        hp.print_classification_percentage(class_type[0], percentage_correct)

    print(f'---> {total_preccision/len(all_training_hist)} average preccision.')
    print(hp.LONG_LOCOMOTIVE)


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

    total_preccision = 0
    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            label = hp.k_NN(hist, all_training_hist, dist_func=hp.euclidean_distance, k=k)
            images_and_labels[class_type].append((label[0], test_images_dir[class_type][amount]))
            if label[0] == class_type[0]:
                correct_label += 1
            amount += 1
        percentage_correct = int((correct_label / amount) * 100)
        total_preccision += percentage_correct
        hp.print_classification_percentage(class_type[0], percentage_correct)

    print(f'---> {total_preccision/len(all_test_hist)} average preccision.')
    print(hp.LONG_LOCOMOTIVE)

    return images_and_labels

if __name__ == "__main__":
    # label_all_training_images(hp.HISTOGRAM_FILE_EXT)
    # label_all_training_images(hp.HISTOGRAM_SMALL_FILE_EXT)
    # label_all_training_images(hp.HISTOGRAM_EUCLIDEAN_FILE_EXT)
    # label_all_training_images(hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT)

    print("Classification using euclidean distance... \n" + hp.LONG_LOCOMOTIVE)
    result = label_all_test_images(hp.HISTOGRAM_FILE_EXT, k=3)
    result_small_codebook = label_all_test_images(hp.HISTOGRAM_SMALL_FILE_EXT, k=24)
    result_euclidean_codebook = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_FILE_EXT, k=1)
    result_small_euclidean_codebook = label_all_test_images(hp.HISTOGRAM_EUCLIDEAN_SMALL_FILE_EXT, k=23)

    for key in result:
         hp.display_multiple_image_with_labels(key, result[key])
