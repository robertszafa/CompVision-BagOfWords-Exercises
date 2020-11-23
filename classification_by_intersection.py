from typing import Dict, List
import helper as hp
import collections 

################################################################################
# Step 6. Intersection between two histograms
################################################################################

# Same number of bin (i.e. codewords)
def intersection(test_hist, train_hist):
    total = 0
    for codeword_freq in train_hist:
        total += min(test_hist[codeword_freq], train_hist[codeword_freq])
    return total

def apply_intersection(test_hist, training_hist_by_classes, limit=None) -> Dict[str, int]:
    result = collections.defaultdict(int)

    for class_type in training_hist_by_classes:
        n = limit if limit else len(training_hist_by_classes[class_type])
        for hist in range(n):
            train_hist = training_hist_by_classes[class_type][hist]
            result[class_type] += intersection(test_hist, train_hist)
    return result

def label_histogram_by_intersection(test_hist, train_hist, limit=None):
    # Classify one test histogram given multiple training histograms of multiple classes
    result = apply_intersection(test_hist, train_hist, limit)
    label = None

    for key in result:
        if not label:
            label = (key[0], result[key])
        elif result[key] > label[1]:
            label = (key[0], result[key])
    return label

def label_by_intersection(limit=None):
    # Get the paths of all histogram files
    test_path, training_path = hp.get_histogram_paths()

    # Load all histogram binary file name
    all_test_hist = hp.load_all_histograms(test_path)
    all_training_hist = hp.load_all_histograms(training_path)

    for class_type in all_test_hist:
        correct_label, amount = 0, 0
        for hist in all_test_hist[class_type]:
            result = label_histogram_by_intersection(hist, all_training_hist, limit)
            amount += 1
            if result[0] == class_type[0]:
                correct_label += 1
        percentage_correct = int((correct_label / amount) * 100)
        print(f'{class_type[0]} correct label is {percentage_correct}%')