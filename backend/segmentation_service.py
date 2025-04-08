import numpy as np
import cv2
import io
import base64
from PIL import Image
from utils_segment_image import (
    preprocess_image,
    analyze_nuclei,
    create_overlay
)
import openslide
import os
import logging
from utils_segment_image import (
    get_background_mask,
)
from constants import (
    TILE_SIZE,
    TOP_BIGGEST_CONTOURS_TO_OBSERVE,
    CONTOUR_LEVEL,
    MIN_FRACTION_OF_TILE_INSIDE_CONTOUR
)

from utils_segment_whole_slide import (
    get_saved_contours,
    get_top_biggest_contours,
    tile_and_save_contours
)
def process_image_segmentation_request(file: io.BytesIO) -> dict:
    """
    High-level function that takes in the image file and microns per pixel,
    runs the necessary steps, and returns the final dictionary response.
    """
    # TODO ADD A DIFFERENT LOGIC PATH IF THE FILE HAPPENS TO BE AN NDPI FILE
    # Convert to a numpy-friendly format
    gray_array, image = preprocess_image(file)
    original_image = np.array(image)

    # This function call returns the count of nuclei and the two masks.
    total_cells, nuclei_mask, background_mask = analyze_nuclei(original_image, gray_array)

    # Create overlays, encode output, etc.
    overlay_image = create_overlay(original_image, nuclei_mask, np.array([0, 255, 0], dtype=np.uint8))
    overlay_image_background = create_overlay(
        overlay_image,
        background_mask,
        overlay_color=np.array([0, 0, 255], dtype=np.uint8)
    )

    # Convert final image to base64-encoded PNG
    img_byte_arr = io.BytesIO()
    Image.fromarray(overlay_image_background).save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Assemble the response JSON data
    response_data = {
        "total_cell_count": total_cells,
        "segmented_image": img_base64,
        # "foreground_area": ... IN MM SQ. ONLY AVAILABLE IF NDPI
    }

    return response_data

def process_ndpi_segmentation_request(
    slide,
    ndpi_file=None,
    contour_level=CONTOUR_LEVEL,
    top_biggest_contours_to_observe=TOP_BIGGEST_CONTOURS_TO_OBSERVE,
    tile_size=TILE_SIZE,
    min_fraction_of_tile_inside_contour=MIN_FRACTION_OF_TILE_INSIDE_CONTOUR,
):
    """
    Process an OpenSlide object to:
      1) Reopen it if needed.
      2) Extract a downsampled image at 'contour_level'.
      3) Detect and draw contours for visualization.
      4) Generate and save tissue/non-tissue masks.
      5) Tile the regions of interest and save them.
    
    :param slide:                  OpenSlide object
    :param ndpi_file:             File path to the .ndpi image (used if slide is closed)
    :param contour_level:         The pyramid level to use for contour detection
    :param top_biggest_contours_to_observe: 
                                  How many largest contours to keep
    :param tile_size:             Size (width/height) of each tile in pixels
    :param min_fraction_of_tile_inside_contour:
                                  The minimum fraction of a tile that must overlap
                                  with the contour to be saved
    :param output_tiles_path:     Where to write output images/tiles (if not None)
    """

    # If the slide is closed, reopen it
    if slide._osr is None and ndpi_file is not None:
        slide = openslide.OpenSlide(ndpi_file)

    # Read the downsampled image at the requested level
    level_image = slide.read_region((0, 0), contour_level, slide.level_dimensions[contour_level])
    level_array = np.array(level_image.convert("RGB"))

    # Get contours from the array
    saved_contours = get_saved_contours(level_array)
    
    # Create overlay image to visualize contours
    overlay_image = level_array.copy()
      # example fallback

    # Draw the contours on the overlay image
    for contour in saved_contours:
        cv2.drawContours(overlay_image, [contour], -1, (0, 255, 0), 2)

    # Create a temporary buffer to store the overlay image
    temp_buffer = io.BytesIO()
    Image.fromarray(overlay_image).save(temp_buffer, format='PNG')
    temp_buffer.seek(0)
    if output_tiles_path is not None:
        overlay_image_path = os.path.join(output_tiles_path, "overlay_image_with_contours.png")
        os.makedirs(output_tiles_path, exist_ok=True)
        cv2.imwrite(overlay_image_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        logging.info(f"Overlay image with contours saved to: {overlay_image_path}")

    # Create tissue/non-tissue mask
    mask_background = get_background_mask(level_array)      # 255 = background, 0 = tissue
    tissue_mask = cv2.bitwise_not(mask_background)          # invert: 255 = tissue, 0 = background

    # Optionally, regenerate or update contours (for example, if you need only the top largest ones)
    saved_contours = get_saved_contours(level_array)
    saved_contours = get_top_biggest_contours(saved_contours, top_n=top_biggest_contours_to_observe)

    # Tile and save contours
    result = tile_and_save_contours(
        slide,
        tissue_mask=tissue_mask,
        mask_background=mask_background,
        mpp_x=float(slide.properties.get('openslide.mpp-x', 0.0)),
        mpp_y=float(slide.properties.get('openslide.mpp-y', 0.0)),
        saved_contours=saved_contours,
        downsample_factor=slide.level_downsamples[contour_level],
        tile_size=tile_size,
        min_coverage_fraction=min_fraction_of_tile_inside_contour,
        output_dir=output_tiles_path
    )