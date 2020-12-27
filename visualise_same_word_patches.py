"""
CW1-COMP338 - Step 3.3 Visualize some image patches that are assigned to the same codeword.

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import cv2
import argparse

import helper as hp

## Step 3.3
def draw_keypoint(img_fname, kp_x, kp_y, kp_diameter, title=''):
    img = cv2.imread(img_fname)
    radius = kp_diameter//2
    kp = img[kp_x-radius : kp_x+radius, kp_y-radius : kp_y+radius]

    cv2.imshow(title, kp)
    cv2.waitKey(0)

def visualize_similar_patches(map_kps_to_codewords):
    for word_idx, img_fname_keypoints_pairs in enumerate(map_kps_to_codewords):
        for img_fname, kps in img_fname_keypoints_pairs.items():
            for kp in kps:
                title = title=f'{img_fname.split(hp.DATASET_DIR)[1]} --> codeword_{word_idx}'
                draw_keypoint(img_fname, int(kp[0][0]), int(kp[0][1]), int(kp[1]), title)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise image keypoints that were assigned \
                                     to the same code word in the disctionary of visual words.')
    parser.add_argument('-e', help='use codebook generated using euclidean distance', action='store_true')
    parser.add_argument('-s', help='use small codebook', action='store_true')
    args = parser.parse_args()

    if args.e and args.s:
        codebook_file = hp.CODEBOOK_EUCLIDEAN_SMALL_FILE
        map_kp_to_word_file = hp.MAP_KPS_TO_CODEBOOK_EUCLIDEAN_SMALL_FILE
    elif args.e:
        codebook_file = hp.CODEBOOK_EUCLIDEAN_FILE
        map_kp_to_word_file = hp.MAP_KPS_TO_CODEBOOK_EUCLIDEAN_FILE
    elif args.s:
        codebook_file = hp.CODEBOOK_SMALL_FILE
        map_kp_to_word_file = hp.MAP_KPS_TO_CODEBOOK_SMALL_FILE
    else:
        codebook_file = hp.CODEBOOK_FILE
        map_kp_to_word_file = hp.MAP_KPS_TO_CODEBOOK_FILE

    dictionary_dist_func = 'euclidean' if args.e else 'Sum Of Absolute Difference'
    num_words = 20 if args.s else 500
    print(f'---> Dictionary of {num_words} visual words was clusterd using {dictionary_dist_func} distance function')

    codebook = hp.load_pickled_list(codebook_file)
    map_kp_to_words = hp.load_pickled_list(map_kp_to_word_file)

    # Put most matched codewords and the corresponding keypoints to the front.
    num_of_kps = lambda d : sum([len(v) for v in d.values()])
    map_kp_to_words.sort(key=num_of_kps, reverse=True)

    visualize_similar_patches(map_kp_to_words)
