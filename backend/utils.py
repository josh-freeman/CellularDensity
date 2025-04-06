from typing import List, Tuple
import cv2
import numpy as np
import os
import base64

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

def get_mask(image: np.ndarray) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 1)
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
    rb_histogram = histogram[:, 0] + histogram[:, 2]
    cumulative_sum = np.cumsum(rb_histogram)

    # Find the total number of pixels in the R+B channel
    total_pixels = cumulative_sum[-1]

    # Determine the pixel intensity corresponding to the given percentile
    target_value = total_pixels * (percentile / 100.0)
    rb_percentile_value = np.searchsorted(cumulative_sum, target_value)
    return rb_percentile_value
