import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import logging
from utils_segment_image import (
    compute_nuclei_mask_and_count,
    get_background_mask
)
from constants import (
    TOP_BIGGEST_CONTOURS_TO_OBSERVE
)
import openslide
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_saved_contours(level_array):    
    mask_background = get_background_mask(level_array)
    
    # Apply a kernel to smoothen the mask
    kernel = np.ones((3, 3), np.uint8)
    smoothed_mask = cv2.morphologyEx(mask_background, cv2.MORPH_CLOSE, kernel)
    
    #closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    #cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Invert the smoothed mask
    inverted_mask = cv2.bitwise_not(smoothed_mask)

    # Find contours in the inverted mask
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def prepare_tile_info(
    x0: int, y0: int,
    w0: int, h0: int,
    tile_size:  int,
    scale_x: float, scale_y: float
) -> List[Tuple]:
    """
    Prepare a list of tile info tuples for the bounding rect in level-0.
    Each tuple is:
      (row_idx, col_idx,
       tile_start_x, tile_start_y,
       current_tile_w, current_tile_h,
       x8, y8, w8, h8)
    where x8, y8, w8, h8 are the tile coords at level-8.
    """
    n_tiles_x = math.ceil(w0 / tile_size)
    n_tiles_y = math.ceil(h0 / tile_size)

    tile_info_list = []
    def process_tile(row_idx, col_idx):
        tile_start_x = x0 + col_idx * tile_size
        tile_start_y = y0 + row_idx * tile_size

        current_tile_w = min(tile_size, x0 + w0 - tile_start_x)
        current_tile_h = min(tile_size, y0 + h0 - tile_start_y)
        if current_tile_w <= 0 or current_tile_h <= 0:
            return None

        # Level-8 coords for the tile
        x8 = int(tile_start_x / scale_x)
        y8 = int(tile_start_y / scale_y)
        w8 = int(current_tile_w / scale_x)
        h8 = int(current_tile_h / scale_y)

        return (
            row_idx, col_idx,
            tile_start_x, tile_start_y,
            current_tile_w, current_tile_h,
            x8, y8, w8, h8
        )

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_tile, row_idx, col_idx)
            for row_idx in range(n_tiles_y)
            for col_idx in range(n_tiles_x)
        ]
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                tile_info_list.append(result)
    return tile_info_list


def process_one_tile(
    tile_info: Tuple,
    slide,
    # Instead of a blanket tissue_mask, we pass a "contour_mask" that 
    # is already 255 inside the specific polygon and 0 elsewhere at level-8.
    contour_mask: np.ndarray,  
    mask_background: np.ndarray,  # 255 = macro-level background, 0 = tissue
    scale_x: float,
    scale_y: float,
    min_coverage_fraction: float,
) -> dict:
    """
    Process a single tile given in tile_info. Returns a dictionary with:
      {
        "row_idx": ...,
        "col_idx": ...,
        "tile_img": (the final overlayed tile or None if coverage < threshold),
        "nuclei_count": (number of nuclei in tile)
      }
    """
    (row_idx, col_idx,
     tile_start_x, tile_start_y,
     current_tile_w, current_tile_h,
     x8, y8, w8, h8) = tile_info

    max_y8 = contour_mask.shape[0]
    max_x8 = contour_mask.shape[1]
    x8_end = min(x8 + w8, max_x8)
    y8_end = min(y8 + h8, max_y8)

    # If the region is out of bounds in level-8 mask
    if x8_end <= 0 or y8_end <= 0 or x8 >= max_x8 or y8 >= max_y8:
        return {
            "row_idx": row_idx,
            "col_idx": col_idx,
            "tile_img": None,
            "nuclei_count": 0
        }

    # Tissue coverage check against the *polygon-specific mask*
    tile_mask_slice = contour_mask[y8:y8_end, x8:x8_end]
    inside_count = cv2.countNonZero(tile_mask_slice)
    slice_area_8 = (y8_end - y8) * (x8_end - x8)
    coverage_fraction = (inside_count / slice_area_8) if slice_area_8 > 0 else 0.0

    if coverage_fraction < min_coverage_fraction:
        return {
            "row_idx": row_idx,
            "col_idx": col_idx,
            "tile_img": None,
            "nuclei_count": 0
        }

    # Read tile at level 0
    region = slide.read_region(
        (tile_start_x, tile_start_y),
        0,
        (current_tile_w, current_tile_h)
    ).convert("RGB")
    tile_img = np.array(region)

    # 1) Overlay the macro-level background (blue)
    bkg_slice = mask_background[y8:y8_end, x8:x8_end]
    if bkg_slice.size > 0:
        bkg_slice_resized = cv2.resize(
            bkg_slice,
            (current_tile_w, current_tile_h),
            interpolation=cv2.INTER_NEAREST
        )
        # Color background in blue
        tile_img[bkg_slice_resized == 255] = (0, 0, 255)  # RGB

    # 2) Detect nuclei in tile, overlay them in green
    gray_tile = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
    total_cells, nuclei_mask = compute_nuclei_mask_and_count(tile_img, gray_tile)

    # Overwrite tile pixels where nuclei_mask==255 with green
    tile_img[nuclei_mask == 255] = (0, 255, 0)

    return {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "tile_img": tile_img,
        "nuclei_count": total_cells
    }


def create_mosaic(
    tile_grid: List[List[np.ndarray]],
    tile_size: int
) -> np.ndarray:
    """
    Create a single large mosaic image from a 2D grid of tile images (RGB).
    Tiles may be None (no coverage); those appear as black squares.
    TODO this could be parallelized and used as background tasks for e.g. celery
    or similar.
    """
    n_tiles_y = len(tile_grid)
    n_tiles_x = len(tile_grid[0]) if n_tiles_y > 0 else 0

    mosaic_h = n_tiles_y * tile_size
    mosaic_w = n_tiles_x * tile_size

    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for row_idx in range(n_tiles_y):
        for col_idx in range(n_tiles_x):
            tile_img = tile_grid[row_idx][col_idx]
            y0 = row_idx * tile_size
            x0 = col_idx * tile_size
            if tile_img is not None:
                h, w = tile_img.shape[:2]
                mosaic[y0:y0 + h, x0:x0 + w] = tile_img

    return mosaic

def get_top_biggest_contours(contours, top_n=TOP_BIGGEST_CONTOURS_TO_OBSERVE):
    """
    Get the top N biggest contours based on area.
    """
    # Sort contours by area (descending)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Select the top N biggest contours
    top_contours = sorted_contours[:top_n]
    
    return top_contours

def tile_and_save_contours(
    slide: openslide.OpenSlide,
    tissue_mask: np.ndarray,      # 255 = tissue, 0 = background (at level-8)
    mask_background: np.ndarray,  # 255 = background, 0 = tissue (level-8)
    mpp_x: float,
    mpp_y: float,
    saved_contours,
    downsample_factor: float,
    tile_size: int,
    min_coverage_fraction: float = 0.5,
    output_dir: str | None = None
):
    """
    For each polygon contour in `saved_contours` (level-8 coords), do:
      1. Get its bounding box in level-8 coords
      2. Create a polygon-specific mask for coverage checking
      3. Tile that bounding box at level-0 coords
      4. For each tile, overlay macro background + detect/overlay nuclei
      5. Stitch everything into a mosaic and save

    Also logs total cell count per contour and non-background area in mm².
    """

    # Downsample factor for level-8
    scale_x = scale_y = downsample_factor  # typically a float
    total_nuclei_count = 0
    total_non_background_area_mm2 = 0
    for i, contour_pts in enumerate(saved_contours, start=1):
        # 1) Get bounding rect of the contour (x,y,w,h) in level-8 coords
        x, y, w, h = cv2.boundingRect(contour_pts)
        logging.info(f"\n[Contour #{i}] bounding rect (level-8 coords): ({x}, {y}, {w}, {h})")

        # 2) Create a polygon-specific mask (same size as the entire tissue_mask)
        #    so that coverage checks only consider this contour region
        contour_mask = np.zeros_like(tissue_mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour_pts], 
                         contourIdx=-1, 
                         color=255, 
                         thickness=-1)  # filled polygon in white

        # 3) Convert bounding rect to level-0 coords
        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        w0 = int(w * scale_x)
        h0 = int(h * scale_y)
        logging.info(f"[Contour #{i}] bounding rect (level-0 coords): ({x0}, {y0}, {w0}, {h0})")

        # Prepare tile info
        tile_info_list = prepare_tile_info(
            x0, y0, w0, h0,
            tile_size, scale_x, scale_y
        )

        # Process tiles in parallel
        n_tiles_x = math.ceil(w0 / tile_size)
        n_tiles_y = math.ceil(h0 / tile_size)
        tile_grid = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]
        tile_nuclei_counts = [[0 for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]
        total_nuclei_for_contour = 0

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_one_tile,
                    tile_info,
                    slide,
                    contour_mask, 
                    mask_background,
                    scale_x,
                    scale_y,
                    min_coverage_fraction,
                )
                for tile_info in tile_info_list
            ]

            for fut in as_completed(futures):
                result = fut.result()
                r = result["row_idx"]
                c = result["col_idx"]
                tile_img = result["tile_img"]
                nuclei_count = result["nuclei_count"]

                total_nuclei_for_contour += nuclei_count
                tile_grid[r][c] = tile_img
                tile_nuclei_counts[r][c] = nuclei_count

        logging.info(f"[Contour #{i}] total nuclei count: {total_nuclei_for_contour}")

        # 4) Compute area in mm² (non-background area)
        #    Because our polygon is in tissue_mask as well, we can slice that region
        #    if you want area specifically for the polygon region:
        contour_mask_slice = contour_mask[y:y+h, x:x+w]
        # Count how many level-8 pixels are inside the polygon
        non_background_pixels = np.count_nonzero(contour_mask_slice)

        # Each pixel in level-8 might represent (scale_x * scale_y) level-0 pixels
        # and each level-0 pixel is (mpp_x * mpp_y) µm² => so per-pixel in level-8 
        # is (scale_x * scale_y * mpp_x * mpp_y) µm² => convert to mm² => multiply by 1e-6
        non_background_area_mm2 = non_background_pixels *  scale_x * scale_y * mpp_x * mpp_y * 1e-6

        logging.info(f"[Contour #{i}] Non-background area: {non_background_area_mm2:.2f} mm²")

        # 5) Create the mosaic and save
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            puzzle_mosaic = create_mosaic(tile_grid, tile_size)
            out_path = os.path.join(output_dir, f"contour_{i:03d}_puzzle.png")
            Image.fromarray(puzzle_mosaic).save(out_path)
            logging.info(f"[Contour #{i}] Mosaic saved to: {out_path}")
        total_nuclei_count += total_nuclei_for_contour
        total_non_background_area_mm2 += non_background_area_mm2

    return \
        {
            "total_nuclei_count": total_nuclei_count,
            "total_non_background_area_mm2": total_non_background_area_mm2
        }
