
import helper as hp
import scipy.cluster.vq as vq
import random

from histogram_of_codewords import gen_histograms
from classification_by_euclidean import *


if __name__ == "__main__":
    tmp = hp.load_descriptors(test_or_train='Training', merge_in_class=True)
    min_len_descriptors = min(map(len, tmp.values()))
    all_descriptors = []
    for c, descriptors in tmp.items():
        all_descriptors += descriptors
        # all_descriptors += random.sample(descriptors, min_len_descriptors)
        print(c, ' ', len(descriptors))


    training_descriptors = hp.load_descriptors(test_or_train='Training', merge_in_class=False)
    test_descriptors = hp.load_descriptors(test_or_train='Test', merge_in_class=False)

    # # Note that keypoint i of a given image corresponds to descriptor i of that image.
    training_keypoints = hp.load_keypoints(test_or_train='Training', merge_in_class=False)
    test_keypoints = hp.load_keypoints(test_or_train='Test', merge_in_class=False)

    CODEBOOK_PYSIFT = f'{hp.DATASET_DIR}/Training/codebook_gpu.npy'
    codebook, _ = vq.kmeans(all_descriptors, 500)
    hp.save_to_pickle(CODEBOOK_PYSIFT, codebook)

    map_kps_to_codebook = gen_histograms(training_descriptors, test_descriptors,
                                         training_keypoints, test_keypoints,
                                         codebook, hist_file_extension='_histogram_gpu.npy')

    label_all_test_images_v2()
