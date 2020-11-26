# -*- coding: utf-8 -*-
"""
CW1-COMP338 - Step 1. Feature extraction

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211

"""

import cv2
import math
import time
import os
import numpy as np
from functools import cmp_to_key

from helper import load_images_in_directory, DATASET_DIR, CLASSES


################################################################################
# Helper methonds specifically for SIFT.
################################################################################
def convolution(img, kernel, average=False):
    """
    Convolute an image with a kernel. If the passed in image has 3 colour channels,
    then convert it to grayscale.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape

    # Zero-padd the img beyond borders.
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img

    # Slide over the img and apply the kernel.
    output = np.zeros(img.shape)
    for row in range(img_row):
        for col in range(img_col):
            # Convoultion:
            # 1. Get the product of the values of the corresponding pixels
            #    in both the kelnel and the patch of the padded img
            # 2. Sum up all the products to get the output for this specific pixel.
            output[row, col] = np.sum(kernel * padded_img[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output

def get_dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def round_up_to_odd(f):
    """
    Given a floating point number, round it UP to the next odd integer.
    """
    return int(np.ceil(f)//2 * 2 + 1)

def gen_gaussian_kernel(sigma):
    """
    Given a sigma value, generate a 2D Gaussian kernel
    with 3*sigma pixels in each direction.
    """
    # Ensure the kernel has a proper middle by using only odd size.
    size = round_up_to_odd(3 * sigma)
    kernel_1D = np.linspace(-(size//2), size//2, size)

    for i in range(size):
        kernel_1D[i] = get_dnorm(kernel_1D[i], 0, sigma)

    # A 2D Gaussian distribution is just (1D_gaussian, 1D_gaussian)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


################################################################################
# (1) Scale-space extrema detection
################################################################################
def gen_gaussian_sigmas_in_octaves(sigma):
    """
    Given a staring value of sigma, calculate a sequence of sigma values, where
    each octave value corresponds to doubling the value of the previous sigma.
    """
    # Default parameters from Lowe's SIFT paper.
    imgs_per_octave = 6
    k = 2 ** (1.0 / imgs_per_octave)

    gaussian_sigmas = np.zeros(imgs_per_octave)

    gaussian_sigmas[0] = sigma
    for img_idx in range(1, imgs_per_octave):
        sigma_previous = (k ** (img_idx - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_sigmas[img_idx] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)

    return gaussian_sigmas

def apply_gaussian_kernels(img, num_octaves, sigmas):
    """
    Generate scale-space pyramid of Gaussian images.
    """
    gaussian_imgs = []

    for _ in range(num_octaves):
        gaussian_imgs_in_octave = []

         # The sigmas[0] kernel won't change the image, so skip it.
        gaussian_imgs_in_octave.append(img)
        for sigma in sigmas[1:]:
            gaussian_kernel = gen_gaussian_kernel(sigma)
            gaussian_imgs_in_octave.append(convolution(img, gaussian_kernel))

        gaussian_imgs.append(gaussian_imgs_in_octave)
        # Octave base will be in the middle.
        octave_base = gaussian_imgs_in_octave[len(gaussian_imgs_in_octave)//2 - 1]
        img = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                         interpolation=cv2.INTER_NEAREST)

    return np.array(gaussian_imgs, dtype=object)

def do_dog(gaussian_imgs):
    dog_images = []

    for gaussian_imgs_octave in gaussian_imgs:
        dog_octave = []

        # Loop through evry pair in the octave.
        for first_image, second_image in zip(gaussian_imgs_octave, gaussian_imgs_octave[1:]):
            # Use np. subract because we're working with uint8 dtypes.
            dog = np.subtract(second_image, first_image)
            dog_octave.append(dog)

        dog_images.append(dog_octave)

    return np.array(dog_images, dtype=object)


################################################################################
# (2) Keypoint localization
################################################################################
def identify_keypoints(gaussian_images, dog_images):
    """
    Keypoints are identified as local minima/maxima of the DoG images across scales
    – by comparing each pixel in the DoG images to its eight neighbors at the
      same scale and nine corresponding neighboring pixels in each of the neighboring scales
    - comparisons with the nearest 26 neighbours in a discretized scale-space volume
    – if the pixel value is the maximum or minimum among all compared pixels,
      it is selected as a candidate keypoint

    Return as list of KeyPoint OpenCV objects, representing SIFT keypoints.
    """
    # Constants from OpenCV and following Lowe's SIFT paper.
    contrast_threshold = 0.04
    image_border_width = 5
    sigma = 1.6
    num_intervals = 3
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)

    keypoints = []
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        img_tripple = zip(dog_images_in_octave,
                          dog_images_in_octave[1:],
                          dog_images_in_octave[2:])
        for image_index, (first_image, second_image, third_image) in enumerate(img_tripple):

            # (i, j) is the center of the 3x3 array
            # Don't take pixels at the border under consideration
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    # Compare with 27 neighbours, i.e. 9 neighbours from each:
                    # octave_index-1, octave_index, octave_index+1
                    if is_px_extremum(first_image[i-1:i+2, j-1:j+2],
                                      second_image[i-1:i+2, j-1:j+2],
                                      third_image[i-1:i+2, j-1:j+2],
                                      threshold):
                        localization_result = find_extrema(i, j, image_index + 1,
                                                           octave_index, num_intervals,
                                                           dog_images_in_octave, sigma,
                                                           contrast_threshold, image_border_width)

                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientation = assign_orientations(keypoint, octave_index,
                                                                            gaussian_images[octave_index][localized_image_index])

                            keypoints += keypoints_with_orientation

    return keypoints

def is_px_extremum(img1, img2, img3, threshold):
    """
    Return True if the center element of the 3x3x3 input array is the maximum
    or minimum among all compared pixels.
    """
    center_pixel_value = img2[1, 1]

    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= img1) and \
                   np.all(center_pixel_value >= img3) and \
                   np.all(center_pixel_value >= img2[0, :]) and \
                   np.all(center_pixel_value >= img2[2, :]) and \
                   center_pixel_value >= img2[1, 0] and \
                   center_pixel_value >= img2[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= img1) and \
                   np.all(center_pixel_value <= img3) and \
                   np.all(center_pixel_value <= img2[0, :]) and \
                   np.all(center_pixel_value <= img2[2, :]) and \
                   center_pixel_value <= img2[1, 0] and \
                   center_pixel_value <= img2[1, 2]

    return False

def do_gradient(pixel_array):
    """
    Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array.
    """
    # f'(x) = (f(x + h) - f(x - h)) / (2 * h)
    # where h = 1, so f'(x) = (f(x + 1) - f(x - 1)) / 2
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])

    return np.array([dx, dy, ds])

def do_hessian(pixel_array):
    """
    Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array.
    """
    # f''(x) = (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # where h = 1, so f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    center_pixel_value = pixel_array[1, 1, 1]

    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]

    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] -
                  pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] -
                  pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] -
                  pixel_array[0, 2, 1] + pixel_array[0, 0, 1])

    return np.array([[dxx, dxy, dxs],
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])

def find_extrema(i, j, image_index, octave_index, num_intervals, dog_images_in_octave,
                 sigma, contrast_threshold, image_border_width):
    """
    Iteratively refine pixel positions of scale-space extrema via quadratic fit
    around each extremum's neighbors.
    """
    image_shape = dog_images_in_octave[0].shape

    num_attempts_until_convergence = 3
    for attempt_index in range(num_attempts_until_convergence):
        # Normalize into [0, 1] range.
        img1, img2, img3 = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = np.stack([img1[i-1:i+2, j-1:j+2], img2[i-1:i+2, j-1:j+2],
                               img3[i-1:i+2, j-1:j+2]]).astype('float32') / 255.0

        # Get image gradients.
        gradient = do_gradient(pixel_cube)
        hessian = do_hessian(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            # No significant change in intensity
            break

        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))

        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or \
           j < image_border_width or j >= image_shape[1] - image_border_width or \
           image_index < 1 or image_index > num_intervals:
            # Extremum is outside the image.
            return None

    if attempt_index >= num_attempts_until_convergence:
        # No convergence.
        return None

    f_val = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)

    if abs(f_val)*num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)

        eigenvalue_ratio = 10
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Accept this keypoint. Construct an OpenCV KeyPoint object and return.
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index),
                           (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + \
                              int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
             # octave_index + 1 because the input image was doubled
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * \
                            (2 ** (octave_index + 1))
            keypoint.response = abs(f_val)

            return keypoint, image_index

    return


################################################################################
# (3) Orientation assignment
################################################################################
def assign_orientations(keypoint, octave_index, gaussian_image):
    """Compute orientations for each keypoint
    """
    # Constants from the lecture notes.
    radius_factor = 3
    num_bins = 8
    peak_ratio = 0.8
    scale_factor = 1.5

    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]

                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))

                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.0))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] +
                               4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.

    # Find peaks.
    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                                smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]

        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) /
                                       (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360.0 - interpolated_peak_index * 360.0 / num_bins

            if abs(orientation - 360.0) < 1e-7:
                # Deal with float32 ULP errors beyond 360.0
                orientation = 0

            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations


################################################################################
# Duplicate Keypoint removal
################################################################################
def cmp_keypoints(keypoint1, keypoint2):
    """
    Return True if keypoint1 is less than keypoint2
    """
    # The order of these matters. First check x, y coordinates, then size,
    # angle, response and octave. Use the class_id as a tie breaker.
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave

    return keypoint2.class_id - keypoint1.class_id

def remove_duplicates(keypoints):
    """
    Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(cmp_keypoints))
    unique_keypoints = [keypoints[0]]

    # Identical keypoints will be next to each other in the sorted sequence.
    # It is enough to compare 'next_keypoint' to the previous keypoint.
    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)

    return unique_keypoints


################################################################################
# (4) Keypoint descriptor calculation
################################################################################
def gen_descriptors(keypoints, gaussian_images):
    """
    Generate descriptors for each keypoint
    """
    # Constants from the lecture notes and Lowe's SIFT paper.
    num_bins = 8
    scale_multiplier = 3
    descriptor_max_value = 0.2
    window_width = 4

    descriptors = []
    keypoints_used = []

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape

        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.0
        angle = 360.0 - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)

        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        # First two dimensions are increased by 2 to account for border effects.
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

        # Descriptor window size (described by half_width) follows OpenCV convention.
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        # Ensure half_width lies within image.
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5

                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        # Use local image gradients at the selected scale.
                        dx = gaussian_image[window_row, window_col + 1] - \
                             gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - \
                             gaussian_image[window_row + 1, window_col]

                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))

                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        iter_struct = zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list)
        for row_bin, col_bin, magnitude, orientation_bin in iter_struct:
            # Smoothing via (reverse) trilinear interpolation. Take the center
            # value of the cube and distribute it among its eight neighbors.
            row_bin_floor, col_bin_floor, orientation_bin_floor = \
                np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction = row_bin - row_bin_floor
            col_fraction =  col_bin - col_bin_floor
            orientation_fraction = orientation_bin - orientation_bin_floor

            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        # Remove histogram borders.
        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        # Threshold and normalize descriptor_vector.
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)

        # Multiply by 512, round, and saturate between 0 and 255 to convert from
        # float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255

        if not np.all(descriptor_vector == 0.0):
            descriptors.append(descriptor_vector)
            keypoints_used.append(keypoint)

    return np.array(descriptors, dtype='float32'), keypoints_used

def unpack_octave(keypoint):
    """
    Compute octave, layer, and scale from a keypoint
    """
    # Put octave back into int8 range
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255

    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)

    return octave, layer, scale


################################################################################
# SIFT main function
################################################################################
def extract_SIFT_features(img, sigma=1.6):
    """
    Extract the SIFT features from the input image.

    Returns the extracted keypoints and the generated SIFT descriptors.
    One keypoint is mapped to one descriptor.
    """
    # SIFT uses only a monochrome intensity image.
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # An octave corresponds to doubling the value of sigma.
    # There will be log_2(n) such doublings.
    num_octaves = int(round(math.log(min(img.shape), 2) - 1))

    # Gaussian image pyramid.
    sigmas = gen_gaussian_sigmas_in_octaves(sigma)
    gaussian_images = apply_gaussian_kernels(img, num_octaves, sigmas)
    dog_images = do_dog(gaussian_images)

    keypoints = identify_keypoints(gaussian_images, dog_images)
    keypoints = remove_duplicates(keypoints)

    descriptors, keypoints_used = gen_descriptors(keypoints, gaussian_images)

    return descriptors, keypoints_used


if __name__ == "__main__":
    start_time = time.time()

    # Extract SIFT descriptors from training and test images. Store the descriptors in seperate
    # binary files based on type and class, e.g. trainig_descriptors/cars, test_descriptors/cars, ..
    for training_or_test in ['Training', 'Test']:
        for class_name in CLASSES:
            class_imgs = load_images_in_directory(f'{DATASET_DIR}/{training_or_test}/{class_name}')
            for fname, img in class_imgs.items():
                descriptors, keypoints = extract_SIFT_features(img)

                # Store a single keypoint as [(x, y), diameter].
                # keypoints[i] corresponds to descriptors[i]
                keypoints = [[k.pt, k.size] for k in keypoints]

                fname = fname.split('.')[0]
                d_file = f'{DATASET_DIR}/{training_or_test.lower()}/{class_name}/{fname}_descriptors.npy'
                k_file = f'{DATASET_DIR}/{training_or_test.lower()}/{class_name}/{fname}_keypoints.npy'
                with open(d_file, 'wb') as f:
                    np.save(f, descriptors)
                with open(k_file, 'wb') as f:
                    np.save(f, keypoints)

                print(f'Finished {fname} of {class_name} of {training_or_test} at minute {(time.time() - start_time)//60}')


    print(f'Finished all in {(time.time() - start_time)//60} minutes.')
