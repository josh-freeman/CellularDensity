from typing import List, Tuple
import cv2
import numpy as np
import os
import base64
from concurrent.futures import ThreadPoolExecutor
from constants import (
    KERNEL_SIZE,
    DILATION_ITERATIONS,
    ALPHA_OVERLAY,
    RPB_THRESHOLD_PERCENTILE,
    GRABCUT_ITERATIONS, 
    THRES_PARAMETER
)

def create_overlay(original_image, mask, overlay_color):
    overlay = original_image.copy()
    overlay[mask == 255] = ((1 - ALPHA_OVERLAY) * overlay[mask == 255] + ALPHA_OVERLAY * overlay_color).astype(np.uint8)
    return overlay

def preprocess_image(uploaded_file):
    """Preprocess the image by converting it to grayscale."""
    # Read the image from the uploaded file
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Convert to RGB (H&E is typically in RGB, OpenCV loads as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray, image

def get_background_mask(image: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask of the background in the input image.
    The background is assumed to be the most uniform or dominant region.

    Parameters:
        image (numpy.ndarray): Input BGR image.

    Returns:
        mask (numpy.ndarray): Binary mask where background is 255 and foreground is 0.
    """
    # Create an initial mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define a rectangle that likely contains the foreground (center 90% of image)
    height, width = image.shape[:2]
    rect = (int(0.05 * width), int(0.05 * height), int(0.9 * width), int(0.9 * height))

    # Allocate memory for models (required by GrabCut but not used by you)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Ensure the image is in 3-channel format (CV_8UC3)
    if len(image.shape) == 2 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply GrabCut: an iterative algorithm to segment the image
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)

    # Extract background: 0 and 2 are background pixels
    background_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 255, 0).astype('uint8')

    return background_mask

def get_nuclei_mask(image: np.ndarray) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, THRES_PARAMETER, 1)
    # in order: source image, threshold value, max value, threshold type
    # max value means the maximum intensity value that can be assigned to a pixel
    # threshold type is the type of thresholding operation: 2 = binary inverse, 3 = binary, 4 = truncation, 5 = to zero, 6 = to zero inverse
    
    # Use morphological operations to clean up small noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove small noise
    
    # Find contours of the segmented regions (assumed nuclei)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    # Create an empty mask to display only the nuclei contours
    nuclei_mask = np.zeros_like(thresh)
    cv2.drawContours(nuclei_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return nuclei_mask


def get_map_white_pixels_to_respresentatives(mask: np.ndarray) -> List[np.ndarray]:
    num_labels, labels = cv2.connectedComponents(mask)
    zones = []
    # Calculate the size of each zone
    zone_sizes = [len(np.argwhere(labels == label)) for label in range(1, num_labels)]

    # Save the histogram of zone sizes
    histogram, bin_edges = np.histogram(zone_sizes, bins=20)  # Adjust bins as needed
    histogram_filename = os.path.join("../data", "zone_size_histogram.csv")
    os.makedirs(os.path.dirname(histogram_filename), exist_ok=True)
    np.savetxt(histogram_filename, np.column_stack((bin_edges[:-1], histogram)), delimiter=",", fmt="%d", header="Bin Start,Count", comments="")
    for label in range(1, num_labels):  # Skip background (label 0)
        coords = np.argwhere(labels == label)
        zones.append(coords)  # Each is an (N, 2) array
    return zones

def percentile(vals, p):
    n = sum(vals)
    current_sum = 0
    cutoff = n * p
    for (i, val) in enumerate(vals):
        current_sum += val
        if current_sum >= cutoff:
            return i
    return len(vals) - 1

def calculate_and_save_histogram_and_return_R_cutoff(percentile:float, image: np.ndarray, white_pixel_groups: List[List[Tuple[int, int]]], output_dir: str = "../data"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    #TODO : apply OTSU to get more precise threshold
    # Convert the grayscale image back to RGB
    if len(image.shape) == 2:  # Check if the image is grayscale
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = image  # If already RGB, use the image as is

    # Initialize the histogram
    histogram = np.zeros((256, 3), dtype=int)

    # Calculate the average RGB values for each group
    for group in white_pixel_groups:
        total_rgb = np.zeros(3, dtype=int)
        for x, y in group:
            total_rgb += rgb_image[x, y]
        avg_rgb = total_rgb // len(group)
        for i, value in enumerate(avg_rgb):
            histogram[value, i] += 1

    # save the histogram to a file
    histogram_filename = os.path.join(output_dir, "histogram.csv")
    image_filename = os.path.join(output_dir, "processed_image.png")
    cv2.imwrite(image_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    np.savetxt(histogram_filename, histogram, delimiter=",", fmt="%d")
    # Calculate the cumulative sum of the histogram for the R+B channel
    percentile_otsu = otsu_threshold(histogram)
    rb_histogram = histogram[:, 0] + histogram[:, 2]
    cumulative_sum = np.cumsum(rb_histogram)

    # Find the total number of pixels in the R+B channel
    total_pixels = cumulative_sum[-1]

    # Determine the pixel intensity corresponding to the given percentile
    target_value = total_pixels * (percentile_otsu / 100.0)
    rb_percentile_value = np.searchsorted(cumulative_sum, target_value)
    return rb_percentile_value

def compute_nuclei_mask_and_count(original_image, gray_array):
    """
    Extracted function that computes:
      - The number of nuclei
      - The final nuclei mask (new_mask)

    Does NOT return background_mask.
    """
    # Initial mask and dilation
    mask = get_nuclei_mask(gray_array)
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=DILATION_ITERATIONS)

    # Identify all “zones” of interest
    zones = get_map_white_pixels_to_respresentatives(dilated_mask)

    # Calculate threshold
    rpb_thresh = calculate_and_save_histogram_and_return_R_cutoff(
        RPB_THRESHOLD_PERCENTILE,
        original_image,
        zones
    )

    # Build the final mask
    new_mask = np.zeros_like(mask)
    unfiltered_zones_count = 0

    def process_zone(zone):
        total_rpb = sum(int(original_image[x, y, 0]) for x, y in zone)
        avg_rpb = total_rpb / len(zone)
        if avg_rpb <= rpb_thresh:
            return zone
        return None

    # Parallel zone processing
    with ThreadPoolExecutor() as executor:
        filtered_zones = list(executor.map(process_zone, zones))

    for zone in filtered_zones:
        if zone is not None:
            unfiltered_zones_count += 1
            for x, y in zone:
                new_mask[x, y] = 255

    return unfiltered_zones_count, new_mask


def analyze_nuclei(original_image, gray_array):
    """
    Function that takes in the original image (and, if needed, its grayscale version)
    and returns:
        1) the number of nuclei,
        2) the final nuclei mask,
        3) the background mask.
    """
    # Background mask
    background_mask = get_background_mask(gray_array)

    # Use the newly-extracted helper
    unfiltered_zones_count, new_mask = compute_nuclei_mask_and_count(
        original_image, gray_array
    )

    # Return all three
    return unfiltered_zones_count, new_mask, background_mask

def otsu_threshold(histogram: np.ndarray) -> float:
    """
    Calculate the optimal threshold using Otsu's method.

    Parameters:
        histogram (numpy.ndarray): The RGB histogram (256x3 array).

    Returns:
        float: The optimal threshold value.
    """
    # Combine R and B channels
    rb_histogram = histogram[:, 0] + histogram[:, 2]

    # Total number of pixels
    total_pixels = np.sum(rb_histogram)

    # Initialize variables
    sum_total = np.sum([i * rb_histogram[i] for i in range(256)])
    sum_background = 0
    weight_background = 0
    weight_foreground = 0
    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_background += rb_histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * rb_histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Calculate between-class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # Check if new maximum found
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    percentile_threshold = (threshold / 255) * 100
    return percentile_threshold
