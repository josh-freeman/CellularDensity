#!/usr/bin/env python3
"""
Skin Segmentation Inference Script
==================================

Complete inference script for skin histopathology segmentation models.
Supports batch processing, visualization, and quantitative analysis.
Auto-downloads models from HuggingFace repository: JoshuaFreeman/skin_seg

Key Features:
- EXACT same preprocessing and model architecture as training pipeline
- Auto-detects backbone architecture from model names  
- Downloads any model from HuggingFace JoshuaFreeman/skin_seg repository
- Supports GigaPath, DINOv2, and EfficientNet architectures
- Generates 3 key tissue masks: Epidermis, Dermis (RET+PAP), Structural (GLD+KER+HYP)
- Comprehensive 6-panel visualizations with statistics
- Individual binary mask export
- Batch processing support
- WSI processing with nuclei detection and segmentation
- Experimental connected component voting for nuclei overlay

Usage Examples:
    # List available models:
    python skin_seg_inference.py --list_models
    
    # Use any HuggingFace model:
    python skin_seg_inference.py image.jpg --model_name efficientnet-b3_10x
    python skin_seg_inference.py image.jpg --model_name efficientnet-b5
    python skin_seg_inference.py image.jpg --model_name efficientnet-b7_10x
    python skin_seg_inference.py image.jpg --model_name gigapath
    
    # Use local model:
    python skin_seg_inference.py image.jpg --model_path ./my_model.pt
    
    # Batch processing:
    python skin_seg_inference.py /path/to/images/ --batch --model_name efficientnet-b3_10x
    
    # WSI processing with custom neighborhood size:
    python skin_seg_inference.py slide.ndpi --model_name efficientnet-b3_10x --neighborhood_size 5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import segmentation_models_pytorch as smp

# Try to import OpenSlide for whole slide image support
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Default neighborhood size for adaptive thresholding (same as THRES_PARAMETER in constants.py)
NEIGHBORHOOD_SIZE = 5

def apply_connected_component_voting(nuclei_binary: np.ndarray, non_grayscale_mask: np.ndarray) -> np.ndarray:
    """
    EXPERIMENTAL: Apply connected component voting for nuclei overlay.
    
    Args:
        nuclei_binary: Binary mask of all detected nuclei
        non_grayscale_mask: Mask of non-grayscale (colored/skin) areas
        
    Returns:
        Binary mask of nuclei that should be colored green
    """
    import cv2
    
    # Find connected components in the nuclei mask
    num_labels, labels = cv2.connectedComponents(nuclei_binary.astype(np.uint8))
    
    final_mask = np.zeros_like(nuclei_binary, dtype=bool)
    
    # Check each connected component
    for label in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == label)
        
        # Check if at least one pixel in this component overlaps with non-grayscale area
        has_overlap = np.any(component_mask & non_grayscale_mask)
        
        if has_overlap:
            # Keep the entire connected component
            final_mask |= component_mask
    
    return final_mask

# Color map for 12-class segmentation
COLOR_MAP = {
    0: (108, 0, 115),    # GLD - Gland
    1: (145, 1, 122),    # INF - Inflammation  
    2: (216, 47, 148),   # FOL - Follicle
    3: (254, 246, 242),  # HYP - Hypodermis
    4: (181, 9, 130),    # RET - Reticular
    5: (236, 85, 157),   # PAP - Papillary
    6: (73, 0, 106),     # EPI - Epidermis
    7: (248, 123, 168),  # KER - Keratin
    8: (0, 0, 0),        # BKG - Background
    9: (127, 255, 255),  # BCC - Basal Cell Carcinoma
    10: (127, 255, 142), # SCC - Squamous Cell Carcinoma
    11: (255, 127, 127), # IEC - Inflammatory/Epithelial Cells
}

CLASS_NAMES = {
    0: "GLD", 1: "INF", 2: "FOL", 3: "HYP", 4: "RET", 5: "PAP",
    6: "EPI", 7: "KER", 8: "BKG", 9: "BCC", 10: "SCC", 11: "IEC"
}

FULL_CLASS_NAMES = {
    0: "Gland", 1: "Inflammation", 2: "Follicle", 3: "Hypodermis", 
    4: "Reticular", 5: "Papillary", 6: "Epidermis", 7: "Keratin",
    8: "Background", 9: "Basal Cell Carcinoma", 10: "Squamous Cell Carcinoma", 
    11: "Inflammatory/Epithelial Cells"
}

# Whole slide image extensions
WSI_EXTENSIONS = ['.ndpi', '.svs', '.tif', '.tiff', '.vms', '.vmu', '.scn', '.mrxs', '.bif']


def is_whole_slide_image(image_path: str) -> bool:
    """Check if the image is a whole slide image format."""
    ext = Path(image_path).suffix.lower()
    return ext in WSI_EXTENSIONS


def extract_tile_from_wsi(slide, x: int, y: int, tile_size: int = 224, level: int = 0) -> Image.Image:
    """Extract a single tile from a whole slide image."""
    if not OPENSLIDE_AVAILABLE:
        raise ImportError("OpenSlide is required for whole slide image processing. Install with: pip install openslide-python openslide-bin")
    
    # Extract tile
    tile = slide.read_region((x, y), level, (tile_size, tile_size))
    
    # Convert to RGB (remove alpha channel if present)
    tile_rgb = tile.convert('RGB')
    
    return tile_rgb


def get_optimal_level_for_magnification(slide, target_magnification: str) -> int:
    """Get the optimal pyramid level for the target magnification."""
    if not OPENSLIDE_AVAILABLE:
        return 0
    
    # Extract magnification number (e.g., "10x" -> 10)
    if target_magnification == "unknown":
        return 0
    
    try:
        target_mag = float(target_magnification.replace('x', ''))
    except:
        return 0
    
    # Try to get objective power from slide properties
    try:
        objective_power = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 20))
    except:
        objective_power = 20  # Default assumption
    
    # Calculate which level gives us closest to target magnification
    best_level = 0
    best_diff = float('inf')
    
    for level in range(slide.level_count):
        downsample = slide.level_downsamples[level]
        effective_mag = objective_power / downsample
        diff = abs(effective_mag - target_mag)
        
        if diff < best_diff:
            best_diff = diff
            best_level = level
    
    return best_level


def process_one_tile_with_segmentation(
    tile_info: tuple,
    slide,
    contour_mask: np.ndarray,
    mask_background: np.ndarray,
    scale_x: float,
    scale_y: float,
    min_coverage_fraction: float,
    model: 'SkinSegmentationModel',
    skip_nuclei: bool = False,
    neighborhood_size: int = NEIGHBORHOOD_SIZE
) -> dict:
    """
    Process a single tile with skin segmentation model.
    Similar to process_one_tile but uses segmentation instead of nuclei detection.
    """
    import cv2
    
    (row_idx, col_idx,
     tile_start_x, tile_start_y,
     current_tile_w, current_tile_h,
     x8, y8, w8, h8) = tile_info

    max_y8 = contour_mask.shape[0]
    max_x8 = contour_mask.shape[1]
    x8_end = min(x8 + w8, max_x8)
    y8_end = min(y8 + h8, max_y8)

    # If the region is out of bounds
    if x8_end <= 0 or y8_end <= 0 or x8 >= max_x8 or y8 >= max_y8:
        return {
            "row_idx": row_idx,
            "col_idx": col_idx,
            "tile_img": None,
            "stats": {}
        }

    # Tissue coverage check
    tile_mask_slice = contour_mask[y8:y8_end, x8:x8_end]
    inside_count = cv2.countNonZero(tile_mask_slice)
    slice_area_8 = (y8_end - y8) * (x8_end - x8)
    coverage_fraction = (inside_count / slice_area_8) if slice_area_8 > 0 else 0.0

    is_background_tile = coverage_fraction < min_coverage_fraction

    # Read tile at level 0
    region = slide.read_region(
        (tile_start_x, tile_start_y),
        0,
        (current_tile_w, current_tile_h)
    ).convert("RGB")
    
    # Resize to 224x224 for model prediction
    tile_pil = region.resize((224, 224))
    
    # Run segmentation inference
    pred_mask, confidence_map = model.predict(tile_pil)
    
    # Create visualization overlay
    tile_img = np.array(region)
    
    # Run nuclei detection on original resolution tile (unless skipped for speed or background tile)
    if not skip_nuclei and not is_background_tile:
        from utils_segment_image import compute_nuclei_mask_and_count
        gray_tile = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
        total_nuclei_count, nuclei_mask = compute_nuclei_mask_and_count(tile_img, gray_tile)
    else:
        # For background tiles or testing/speed: create dummy nuclei mask
        total_nuclei_count = 0
        nuclei_mask = np.zeros(tile_img.shape[:2], dtype=np.uint8)
    
    # Resize segmentation mask back to original tile size
    if tile_img.shape[:2] != (224, 224):
        from PIL import Image as PILImage
        pred_mask_resized = np.array(PILImage.fromarray(pred_mask.astype(np.uint8)).resize(
            (current_tile_w, current_tile_h), PILImage.Resampling.NEAREST
        ))
    else:
        pred_mask_resized = pred_mask
    
    # DEBUG: Check what classes are actually predicted
    unique_classes = np.unique(pred_mask_resized)
    epi_pixels = np.sum(pred_mask_resized == 6)
    pap_pixels = np.sum(pred_mask_resized == 5) 
    ret_pixels = np.sum(pred_mask_resized == 4)
    
    # Convert non-skin regions to black and white
    skin_structure_mask = (pred_mask_resized == 6) | (pred_mask_resized == 5) | (pred_mask_resized == 4)
    non_skin_mask = ~skin_structure_mask
    
    # Convert background tiles entirely to B&W, or just non-skin areas for tissue tiles
    if is_background_tile:
        # Convert entire tile to B&W
        gray_values = np.dot(tile_img, [0.299, 0.587, 0.114]).astype(np.uint8)
        tile_img = np.stack([gray_values, gray_values, gray_values], axis=2)
    elif np.any(non_skin_mask):
        # Convert only non-skin areas to grayscale for tissue tiles
        gray_values = np.dot(tile_img[non_skin_mask], [0.299, 0.587, 0.114]).astype(np.uint8)
        tile_img[non_skin_mask] = np.stack([gray_values, gray_values, gray_values], axis=1)
    
    # EXPERIMENTAL: Calculate nuclei overlap with skin regions (only for tissue tiles)
    if is_background_tile:
        # Background tiles: no nuclei processing
        skin_nuclei_count = epi_nuclei_count = pap_nuclei_count = ret_nuclei_count = 0
        final_green_mask = np.zeros(tile_img.shape[:2], dtype=bool)
    else:
        # Tissue tiles: process nuclei normally
        nuclei_binary = (nuclei_mask == 255)
        total_nuclei_pixels = np.sum(nuclei_binary)
        
        if total_nuclei_pixels > 0 and total_nuclei_count > 0:
            # Calculate overlap ratios for statistics
            skin_nuclei_pixels = np.sum(nuclei_binary & skin_structure_mask)
            epi_nuclei_pixels = np.sum(nuclei_binary & (pred_mask_resized == 6))
            pap_nuclei_pixels = np.sum(nuclei_binary & (pred_mask_resized == 5)) 
            ret_nuclei_pixels = np.sum(nuclei_binary & (pred_mask_resized == 4))
            
            # Apply ratios to actual nuclei count (like in original wsi processing)
            skin_ratio = skin_nuclei_pixels / total_nuclei_pixels
            epi_ratio = epi_nuclei_pixels / total_nuclei_pixels
            pap_ratio = pap_nuclei_pixels / total_nuclei_pixels
            ret_ratio = ret_nuclei_pixels / total_nuclei_pixels
            
            skin_nuclei_count = int(total_nuclei_count * skin_ratio)
            epi_nuclei_count = int(total_nuclei_count * epi_ratio)
            pap_nuclei_count = int(total_nuclei_count * pap_ratio)
            ret_nuclei_count = int(total_nuclei_count * ret_ratio)
            
            # EXPERIMENTAL: Apply connected component voting for green overlay
            # Step 1: Get all nuclei as potential green pixels
            all_nuclei_binary = nuclei_binary
            
            # Step 2: Create non-grayscale mask (colored/skin areas vs black & white areas)
            # Areas that are NOT black & white are the colored/skin areas
            non_grayscale_mask = skin_structure_mask  # This represents colored skin areas
            
            # Step 3: Apply connected component voting
            final_green_mask = apply_connected_component_voting(all_nuclei_binary, non_grayscale_mask)
            
        else:
            skin_nuclei_count = epi_nuclei_count = pap_nuclei_count = ret_nuclei_count = 0
            final_green_mask = np.zeros(tile_img.shape[:2], dtype=bool)
    
    # Add green overlay using the voted mask
    tile_img[final_green_mask] = (0, 255, 0)  # Green for nuclei
    
    # Calculate tissue stats with nuclei counts
    tile_stats = {
        "epi_nuclei": epi_nuclei_count,
        "pap_nuclei": pap_nuclei_count,
        "ret_nuclei": ret_nuclei_count,
        "total_skin_nuclei": skin_nuclei_count,
        "epi_area_pixels": np.sum(pred_mask_resized == 6),
        "pap_area_pixels": np.sum(pred_mask_resized == 5),
        "ret_area_pixels": np.sum(pred_mask_resized == 4)
    }

    return {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "tile_img": tile_img,
        "stats": tile_stats
    }


def process_wsi_contours_with_segmentation(
    slide,
    tissue_mask: np.ndarray,
    mask_background: np.ndarray,
    saved_contours,
    downsample_factor: float,
    tile_size: int,
    min_coverage_fraction: float,
    output_dir: str,
    model: 'SkinSegmentationModel',
    neighborhood_size: int = NEIGHBORHOOD_SIZE
) -> Dict[str, float]:
    """
    Process WSI contours with skin segmentation and create mosaics.
    Based on tile_and_save_contours but adapted for segmentation.
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from utils_segment_whole_slide import prepare_tile_info, create_mosaic
    import math
    
    scale_x = scale_y = downsample_factor
    all_contour_stats = []
    
    print(f"🔍 Processing {len(saved_contours)} tissue contours with segmentation...")
    
    for i, contour_pts in enumerate(saved_contours, start=1):
        print(f"\n[Contour #{i}/{len(saved_contours)}] Processing...")
        
        # Get bounding rect of the contour
        x, y, w, h = cv2.boundingRect(contour_pts)
        print(f"[Contour #{i}] bounding rect (level-{int(np.log2(downsample_factor))} coords): ({x}, {y}, {w}, {h})")

        # Create polygon-specific mask
        contour_mask = np.zeros_like(tissue_mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour_pts], 
                         contourIdx=-1, 
                         color=255, 
                         thickness=-1)

        # Convert to level-0 coords
        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        w0 = int(w * scale_x)
        h0 = int(h * scale_y)
        print(f"[Contour #{i}] bounding rect (level-0 coords): ({x0}, {y0}, {w0}, {h0})")

        # Prepare tile info
        tile_info_list = prepare_tile_info(x0, y0, w0, h0, tile_size, scale_x, scale_y)

        # Initialize tile grid
        n_tiles_x = math.ceil(w0 / tile_size)
        n_tiles_y = math.ceil(h0 / tile_size)
        tile_grid = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]
        contour_stats = []

        print(f"[Contour #{i}] Processing {len(tile_info_list)} tiles...")

        # Process tiles in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_one_tile_with_segmentation,
                    tile_info,
                    slide,
                    contour_mask,
                    mask_background,
                    scale_x,
                    scale_y,
                    min_coverage_fraction,
                    model,
                    False,  # skip_nuclei
                    neighborhood_size
                )
                for tile_info in tile_info_list
            ]

            processed_count = 0
            for fut in as_completed(futures):
                result = fut.result()
                r = result["row_idx"]
                c = result["col_idx"]
                tile_img = result["tile_img"]
                stats = result["stats"]

                if tile_img is not None:
                    tile_grid[r][c] = tile_img
                    contour_stats.append(stats)
                    processed_count += 1

        print(f"[Contour #{i}] Processed {processed_count} valid tiles")

        # Aggregate stats for this contour
        if contour_stats:
            contour_aggregated = {
                "epi_nuclei": sum(stats["epi_nuclei"] for stats in contour_stats),
                "pap_nuclei": sum(stats["pap_nuclei"] for stats in contour_stats),
                "ret_nuclei": sum(stats["ret_nuclei"] for stats in contour_stats),
                "total_skin_nuclei": sum(stats["total_skin_nuclei"] for stats in contour_stats),
                "epi_area_pixels": sum(stats["epi_area_pixels"] for stats in contour_stats),
                "pap_area_pixels": sum(stats["pap_area_pixels"] for stats in contour_stats),
                "ret_area_pixels": sum(stats["ret_area_pixels"] for stats in contour_stats)
            }
            
            print(f"[Contour #{i}] Nuclei counts - EPI: {contour_aggregated['epi_nuclei']}, "
                  f"PAP: {contour_aggregated['pap_nuclei']}, RET: {contour_aggregated['ret_nuclei']}, "
                  f"Total: {contour_aggregated['total_skin_nuclei']}")
            print(f"[Contour #{i}] DEBUG: Processed {len(contour_stats)} tiles with valid stats")
            
            all_contour_stats.append(contour_aggregated)

        # Create and save mosaic
        os.makedirs(output_dir, exist_ok=True)
        puzzle_mosaic = create_mosaic(tile_grid, tile_size)
        out_path = os.path.join(output_dir, f"contour_{i:03d}_segmentation_mosaic.png")
        
        from PIL import Image as PILImage
        PILImage.fromarray(puzzle_mosaic).save(out_path)
        print(f"[Contour #{i}] 🎨 Segmentation mosaic saved to: {out_path}")

    # Aggregate stats across all contours
    if all_contour_stats:
        # Get slide properties for area calculation
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0.25))  # Default to 0.25 if not available
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0.25))
        
        # Calculate total skin area in mm²
        total_skin_area_pixels = sum(stats["epi_area_pixels"] + stats["pap_area_pixels"] + stats["ret_area_pixels"] 
                                   for stats in all_contour_stats)
        # Convert pixels to mm²: (pixels * mpp_x * mpp_y) / (1000^2) since mpp is in micrometers
        total_skin_area_mm2 = (total_skin_area_pixels * mpp_x * mpp_y) / 1_000_000
        
        final_stats = {
            "total_nuclei_count": sum(stats["total_skin_nuclei"] for stats in all_contour_stats),
            "total_non_background_area_mm2": total_skin_area_mm2,
            "epi_nuclei": sum(stats["epi_nuclei"] for stats in all_contour_stats),
            "pap_nuclei": sum(stats["pap_nuclei"] for stats in all_contour_stats),
            "ret_nuclei": sum(stats["ret_nuclei"] for stats in all_contour_stats),
            "total_skin_nuclei": sum(stats["total_skin_nuclei"] for stats in all_contour_stats),
            "dermis_nuclei": sum(stats["pap_nuclei"] + stats["ret_nuclei"] for stats in all_contour_stats),
            "Epithelial_total": 0.0,  # Will be calculated as percentage below
            "Dermis_total": 0.0,
            "Cancer_total": 0.0  # Always 0 for skin structure analysis
        }
        
        # Calculate percentages based on nuclei counts
        if final_stats["total_skin_nuclei"] > 0:
            final_stats["Epithelial_total"] = (final_stats["epi_nuclei"] / final_stats["total_skin_nuclei"]) * 100
            final_stats["Dermis_total"] = (final_stats["dermis_nuclei"] / final_stats["total_skin_nuclei"]) * 100
        
        print(f"\n✅ Processed {len(saved_contours)} contours with segmentation mosaics")
        print(f"📊 Total nuclei counts across all contours:")
        print(f"   EPI: {final_stats['epi_nuclei']}")
        print(f"   PAP: {final_stats['pap_nuclei']}")  
        print(f"   RET: {final_stats['ret_nuclei']}")
        print(f"   Total skin nuclei: {final_stats['total_skin_nuclei']}")
        print(f"📏 Total skin area: {final_stats['total_non_background_area_mm2']:.2f} mm²")
        if final_stats['total_non_background_area_mm2'] > 0:
            density = final_stats['total_nuclei_count'] / final_stats['total_non_background_area_mm2']
            print(f"🔬 Nuclei density: {density:.2f} nuclei/mm²")
        
        return final_stats
    else:
        print("❌ No valid tiles processed")
        return {}


def process_wsi_with_tiles(
    image_path: str,
    model: 'SkinSegmentationModel',
    tile_size: int = 224,
    target_magnification: str = "10x",
    output_dir: str = "results",
    neighborhood_size: int = NEIGHBORHOOD_SIZE
) -> Dict[str, float]:
    """
    Process whole slide image using existing WSI infrastructure with proper mosaic creation.
    
    Returns:
        Dict with aggregated statistics
    """
    if not OPENSLIDE_AVAILABLE:
        raise ImportError("OpenSlide is required for whole slide image processing. Install with: pip install openslide-python openslide-bin")
    
    print(f"🔍 Opening whole slide image: {image_path}")
    
    try:
        slide = openslide.OpenSlide(image_path)
    except Exception as e:
        print(f"❌ Error opening WSI with OpenSlide: {e}")
        return {}
    
    # Get slide info
    dimensions = slide.dimensions
    level_count = slide.level_count
    
    print(f"📏 Slide dimensions: {dimensions}")
    print(f"📊 Available levels: {level_count}")
    
    # Use existing WSI infrastructure - import required functions
    from utils_segment_whole_slide import get_saved_contours, get_top_biggest_contours
    from utils_segment_image import get_background_mask
    from constants import CONTOUR_LEVEL, TOP_BIGGEST_CONTOURS_TO_OBSERVE, MIN_FRACTION_OF_TILE_INSIDE_CONTOUR
    import cv2
    
    # Get optimal level for contour detection (typically level 8 or similar)
    contour_level = min(CONTOUR_LEVEL, slide.level_count - 1)
    
    # Read downsampled image for contour detection
    level_image = slide.read_region((0, 0), contour_level, slide.level_dimensions[contour_level])
    level_array = np.array(level_image.convert("RGB"))
    
    print(f"🔍 Using level {contour_level} for tissue detection (dimensions: {slide.level_dimensions[contour_level]})")
    
    # Get tissue contours using existing infrastructure
    saved_contours = get_saved_contours(level_array)
    print(f"🔍 Found {len(saved_contours)} tissue contours")
    
    if not saved_contours:
        print("❌ No tissue contours found")
        slide.close()
        return {}
    
    # Get top biggest contours
    saved_contours = get_top_biggest_contours(saved_contours, top_n=TOP_BIGGEST_CONTOURS_TO_OBSERVE)
    print(f"🎯 Processing top {len(saved_contours)} contours")
    
    # Create tissue/background masks
    mask_background = get_background_mask(level_array)
    tissue_mask = cv2.bitwise_not(mask_background)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    wsi_output_dir = os.path.join(output_dir, base_name)
    
    # Process contours using the skin segmentation model
    result = process_wsi_contours_with_segmentation(
        slide=slide,
        tissue_mask=tissue_mask,
        mask_background=mask_background,
        saved_contours=saved_contours,
        downsample_factor=slide.level_downsamples[contour_level],
        tile_size=tile_size,
        min_coverage_fraction=MIN_FRACTION_OF_TILE_INSIDE_CONTOUR,
        output_dir=wsi_output_dir,
        model=model,
        neighborhood_size=neighborhood_size
    )
    
    slide.close()
    return result


class SkinSegmentationModel:
    """Wrapper class for skin segmentation models."""
    
    def __init__(self, model_path: str = None, model_name: str = None, backbone: str = None, device: str = "auto", requested_magnification: str = None):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Local path to model weights (optional if model_name provided)
            model_name: HuggingFace model name (e.g. "efficientnet-b3", "gigapath", etc.)
            backbone: Backbone architecture (auto-detected from model_name if not provided)
            device: Device to run inference on ("auto", "cuda", or "cpu")
            requested_magnification: Requested magnification (e.g., "10x", "20x") for model selection
        """
        self.device = self._get_device(device)
        self._requested_magnification = requested_magnification
        
        # Auto-detect backbone from model name/path if not provided
        if backbone is None:
            backbone = self._detect_backbone(model_path, model_name)
        
        self.backbone = backbone
        
        # Auto-detect magnification from model name/path
        self.magnification = self._detect_magnification(model_path, model_name)
        if self.magnification != "unknown":
            print(f"🔍 Detected model magnification: {self.magnification}")
        
        # Get model path (download from HF if needed)
        if model_path is None and model_name is not None:
            model_path = self._download_from_hf(model_name, requested_magnification)
        elif model_path is None:
            raise ValueError("Either model_path or model_name must be provided")
            
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _detect_backbone(self, model_path: str = None, model_name: str = None) -> str:
        """Auto-detect backbone architecture from model name or path."""
        source = model_name or model_path or ""
        source = source.lower()
        
        # Check for specific backbones
        if "efficientnet-b3" in source or "efficientnet_b3" in source:
            return "efficientnet-b3"
        elif "efficientnet-b5" in source or "efficientnet_b5" in source:
            return "efficientnet-b5"
        elif "efficientnet-b7" in source or "efficientnet_b7" in source:
            return "efficientnet-b7"
        elif "resnet50" in source:
            return "resnet50"
        elif "resnet34" in source:
            return "resnet34"
        elif "gigapath" in source:
            return "gigapath_vitl"
        elif "dinov2" in source:
            if "base" in source:
                return "vit_base_patch14_dinov2"
            elif "large" in source:
                return "vit_large_patch14_dinov2"
            elif "small" in source:
                return "vit_small_patch14_dinov2"
            else:
                return "vit_base_patch14_dinov2"  # default
        else:
            # Default fallback
            print(f"⚠️  Could not auto-detect backbone from '{source}', using efficientnet-b3")
            return "efficientnet-b3"
    
    def _detect_magnification(self, model_path: str = None, model_name: str = None) -> str:
        """Auto-detect magnification from model name or path."""
        source = model_name or model_path or ""
        source = source.lower()
        
        # Check for magnification patterns
        for mag in ["1x", "2x", "5x", "10x", "20x", "40x"]:
            if mag in source:
                return mag
        
        # Default if not found
        return "unknown"
    
    def _download_from_hf(self, model_name: str, requested_magnification: str = None) -> str:
        """Download model from HuggingFace repository with strict magnification matching."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # List available files in the repository
            try:
                repo_files = list_repo_files("JoshuaFreeman/skin_seg")
                model_files = [f for f in repo_files if f.endswith('.pt')]
                print(f"📋 Available models in JoshuaFreeman/skin_seg: {model_files}")
            except Exception as e:
                print(f"⚠️  Could not list repo files: {e}")
                model_files = []
            
            # Try to find exact match first
            target_file = None
            for pattern in [f"{model_name}_unet_best.pt", f"{model_name}.pt", model_name]:
                if pattern in model_files:
                    target_file = pattern
                    break
            
            # If no exact match, try partial matching
            if target_file is None:
                for file in model_files:
                    if model_name.lower() in file.lower():
                        target_file = file
                        break
            
            # Strict magnification-aware fallback - NO silent failures
            if target_file is None:
                # Extract backbone name from model_name
                backbone_name = model_name.lower()
                if "efficientnet-b3" in backbone_name or "b3" in backbone_name:
                    backbone_base = "efficientnet-b3"
                elif "efficientnet-b5" in backbone_name or "b5" in backbone_name:
                    backbone_base = "efficientnet-b5"
                elif "efficientnet-b7" in backbone_name or "b7" in backbone_name:
                    backbone_base = "efficientnet-b7"
                elif "gigapath" in backbone_name:
                    backbone_base = "gigapath"
                else:
                    backbone_base = "efficientnet-b3"  # default
                
                # Try to find model with requested magnification first
                if requested_magnification:
                    magnification_pattern = f"{backbone_base}_{requested_magnification}_unet_best.pt"
                    if magnification_pattern in model_files:
                        target_file = magnification_pattern
                        print(f"📎 Found exact magnification match: {target_file}")
                
                # If no magnification-specific match, try auto-detected magnification
                if target_file is None and self.magnification != "unknown":
                    magnification_pattern = f"{backbone_base}_{self.magnification}_unet_best.pt"
                    if magnification_pattern in model_files:
                        target_file = magnification_pattern
                        print(f"📎 Found model with auto-detected magnification: {target_file}")
                
                # STRICT: Only try available magnification-specific models for this backbone
                if target_file is None:
                    magnification_patterns = ["10x", "20x", "5x", "2x", "1x", "40x"]  # prioritize common ones
                    for mag in magnification_patterns:
                        magnification_pattern = f"{backbone_base}_{mag}_unet_best.pt"
                        if magnification_pattern in model_files:
                            target_file = magnification_pattern
                            print(f"📎 Found available magnification model: {target_file}")
                            break
                
                # Final fallback - ONLY to properly named models with magnification
                if target_file is None:
                    if backbone_base == "efficientnet-b3":
                        target_file = "efficientnet-b3_10x_unet_best.pt"
                    elif backbone_base == "efficientnet-b5":
                        target_file = "efficientnet-b5_10x_unet_best.pt"  # Assume 10x if b5 gets magnification
                    elif backbone_base == "efficientnet-b7":
                        target_file = "efficientnet-b7_10x_unet_best.pt"
                    elif backbone_base == "gigapath":
                        target_file = "gigapath_10x_unet_best.pt"  # Assume 10x for consistency
                    else:
                        target_file = "efficientnet-b3_10x_unet_best.pt"
            
            print(f"📥 Downloading {target_file} from JoshuaFreeman/skin_seg...")
            
            model_path = hf_hub_download(
                repo_id="JoshuaFreeman/skin_seg",
                filename=target_file,
                cache_dir="./models"
            )
            
            print(f"✅ Downloaded model: {model_path}")
            return model_path
            
        except ImportError:
            print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        except Exception as e:
            print(f"❌ Error downloading model '{model_name}': {e}")
            raise
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the segmentation model with EXACT same architecture as training."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture EXACTLY like training
        if self.backbone == "gigapath_vitl":
            # Use custom GigaPath implementation (same as training)
            try:
                import timm
                class GigaPathSeg(torch.nn.Module):
                    def __init__(self, n_classes=12, pretrained=False):
                        super().__init__()
                        self.backbone = timm.create_model(
                            "hf_hub:prov-gigapath/prov-gigapath",
                            pretrained=pretrained,
                            num_classes=0
                        )
                        
                        # Get patch size
                        _p = self.backbone.patch_embed.patch_size
                        self.patch_h, self.patch_w = (_p, _p) if isinstance(_p, int) else _p
                        C = self.backbone.embed_dim  # 1024
                        
                        self.decoder = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(C, 512, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(512, 256, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(256, 128, 2, stride=2), torch.nn.GELU(),
                            torch.nn.Conv2d(128, n_classes, 1)
                        )
                    
                    def forward(self, x):
                        B, _, H, W = x.shape
                        tokens = self.backbone.forward_features(x)  # B, N+1, C
                        tokens = tokens[:, 1:, :]  # drop CLS
                        
                        # Reshape sequence back to 2-D grid
                        h = H // self.patch_h
                        w = W // self.patch_w
                        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)  # B, C, h, w
                        
                        mask = self.decoder(feat)
                        return torch.nn.functional.interpolate(
                            mask, size=(H, W), mode="bilinear", align_corners=False
                        )
                
                model = GigaPathSeg(n_classes=12, pretrained=False)
                print(f"✅ Created custom GigaPath model")
                
            except Exception as e:
                print(f"❌ Error creating GigaPath model: {e}")
                raise
                
        elif 'dinov2' in self.backbone:
            # Use custom DINOv2 implementation (same as training)
            try:
                import timm
                class DINOv2Seg(torch.nn.Module):
                    def __init__(self, model_name="vit_base_patch14_dinov2", n_classes=12, pretrained=False):
                        super().__init__()
                        self.backbone = timm.create_model(
                            model_name,
                            pretrained=pretrained,
                            num_classes=0,
                            img_size=224
                        )
                        
                        self.patch_h = self.patch_w = self.backbone.patch_embed.patch_size[0]
                        C = self.backbone.embed_dim
                        
                        self.decoder = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(C, 512, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(512, 256, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(256, 128, 2, stride=2), torch.nn.GELU(),
                            torch.nn.Conv2d(128, n_classes, 1)
                        )
                    
                    def forward(self, x):
                        B, _, H, W = x.shape
                        tokens = self.backbone.forward_features(x)  # B, N+1, C
                        tokens = tokens[:, 1:, :]  # drop CLS
                        
                        h = H // self.patch_h
                        w = W // self.patch_w
                        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)  # B, C, h, w
                        
                        mask = self.decoder(feat)
                        return torch.nn.functional.interpolate(
                            mask, size=(H, W), mode="bilinear", align_corners=False
                        )
                
                model = DINOv2Seg(model_name=self.backbone, n_classes=12, pretrained=False)
                print(f"✅ Created custom DINOv2 model")
                
            except Exception as e:
                print(f"❌ Error creating DINOv2 model: {e}")
                raise
        else:
            # Use segmentation-models-pytorch for standard backbones
            model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights=None,  # We'll load our fine-tuned weights
                classes=12,
                activation=None
            )
            print(f"✅ Created SMP model with {self.backbone} backbone")
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Loaded {self.backbone} model from {model_path}")
        return model
    
    def _get_transform(self):
        """Get the preprocessing transform - EXACT same as training pipeline."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Use EXACT same preprocessing as training (no augmentation)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ])
    
    def predict(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation prediction on an image using EXACT same preprocessing as training.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Tuple of (prediction_mask, confidence_map)
        """
        # Convert to numpy array (uint8 0-255, same as training)
        img_array = np.array(image.convert('RGB'))
        h, w = img_array.shape[:2]
        
        # Handle image resizing to 224x224 EXACTLY like training
        patch_size = 224
        if h > patch_size or w > patch_size:
            # Select center patch (same as training inference)
            if h > patch_size:
                y = (h - patch_size) // 2
            else:
                y = 0
            if w > patch_size:
                x = (w - patch_size) // 2
            else:
                x = 0
            img_array = img_array[y:y+patch_size, x:x+patch_size]
        
        # If image is smaller, pad it (same as training inference)
        if img_array.shape[0] < patch_size or img_array.shape[1] < patch_size:
            pad_h = max(0, patch_size - img_array.shape[0])
            pad_w = max(0, patch_size - img_array.shape[1])
            img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
        
        # Apply EXACT same transform as training
        transformed = self.transform(image=img_array)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get prediction and confidence
            confidence, pred_indices = torch.max(probs, dim=1)
            
            pred_mask = pred_indices.squeeze(0).cpu().numpy()
            confidence_map = confidence.squeeze(0).cpu().numpy()
            
        return pred_mask, confidence_map


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB color image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        colored[mask == class_id] = color
    
    return colored


def calculate_tissue_stats(mask: np.ndarray) -> Dict[str, float]:
    """Calculate tissue class statistics."""
    total_pixels = mask.size
    stats = {}
    
    for class_id in range(12):
        count = np.sum(mask == class_id)
        percentage = (count / total_pixels) * 100
        stats[CLASS_NAMES[class_id]] = percentage
    
    # Calculate specialized groupings
    stats["Epithelial_total"] = stats["EPI"]
    stats["Dermis_total"] = stats["RET"] + stats["PAP"] 
    stats["Structural_total"] = stats["GLD"] + stats["KER"] + stats["HYP"]
    stats["Cancer_total"] = stats["BCC"] + stats["SCC"] + stats["IEC"]
    
    return stats


def create_overlay_visualization(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create specialized tissue overlay visualization with key masks."""
    # Convert to numpy for processing
    img_array = np.array(image.resize((mask.shape[1], mask.shape[0])))
    overlay = img_array.copy()
    
    # Create colored overlays with transparency
    alpha = 0.7
    
    # Epidermis (red) - Class 6
    epi_mask = mask == 6
    overlay[epi_mask] = overlay[epi_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    
    # Dermis (blue) - Reticular + Papillary (Classes 4, 5)
    dermis_mask = (mask == 4) | (mask == 5)
    overlay[dermis_mask] = overlay[dermis_mask] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    
    # Structural (green) - Gland + Keratin + Hypodermis (Classes 0, 7, 3)
    struct_mask = (mask == 0) | (mask == 7) | (mask == 3)
    overlay[struct_mask] = overlay[struct_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    
    return Image.fromarray(overlay.astype(np.uint8))


def create_individual_mask_visualization(mask: np.ndarray, original_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create individual binary masks for key tissue types."""
    masks = {}
    
    # Epidermis mask (Class 6)
    masks['epidermis'] = (mask == 6).astype(np.uint8) * 255
    
    # Dermis mask (Classes 4, 5: RET + PAP)
    masks['dermis'] = ((mask == 4) | (mask == 5)).astype(np.uint8) * 255
    
    # Structural mask (Classes 0, 7, 3: GLD + KER + HYP)
    masks['structural'] = ((mask == 0) | (mask == 7) | (mask == 3)).astype(np.uint8) * 255
    
    # Non-structural mask (everything except GLD + KER + HYP)
    masks['non_structural'] = (~((mask == 0) | (mask == 7) | (mask == 3))).astype(np.uint8) * 255
    
    return masks


def create_comprehensive_visualization(
    image: Image.Image,
    pred_mask: np.ndarray,
    confidence_map: np.ndarray,
    stats: Dict[str, float],
    output_path: str
) -> None:
    """Create comprehensive visualization with multiple panels."""
    
    # Resize image to match mask
    image_resized = image.resize((pred_mask.shape[1], pred_mask.shape[0]))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Skin Histopathology Segmentation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Segmentation Result
    colored_mask = mask_to_color(pred_mask)
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('12-Class Segmentation')
    axes[0, 1].axis('off')
    
    # 3. Confidence Map
    conf_plot = axes[0, 2].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('Prediction Confidence')
    axes[0, 2].axis('off')
    plt.colorbar(conf_plot, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. Specialized Overlay
    overlay_img = create_overlay_visualization(image, pred_mask)
    axes[1, 0].imshow(overlay_img)
    axes[1, 0].set_title('Key Tissue Masks\n(Red: Epidermis, Blue: Dermis, Green: Structural)')
    axes[1, 0].axis('off')
    
    # 5. Class Distribution
    class_percentages = [stats[name] for name in CLASS_NAMES.values()]
    colors = [np.array(COLOR_MAP[i]) / 255.0 for i in range(12)]
    
    bars = axes[1, 1].bar(CLASS_NAMES.values(), class_percentages, color=colors)
    axes[1, 1].set_title('Tissue Class Distribution')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, class_percentages):
        if pct > 1:  # Only label if >1%
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 6. Summary Statistics
    axes[1, 2].axis('off')
    summary_text = f"""
KEY TISSUE MASKS:

🔴 EPIDERMIS: {stats['EPI']:.1f}%
   (Class 6 - Epidermis)

🔵 DERMIS: {stats['Dermis_total']:.1f}%
   • Reticular: {stats['RET']:.1f}%
   • Papillary: {stats['PAP']:.1f}%

🟢 STRUCTURAL: {stats['Structural_total']:.1f}%
   • Gland: {stats['GLD']:.1f}%
   • Keratin: {stats['KER']:.1f}%
   • Hypodermis: {stats['HYP']:.1f}%

OTHER CLASSES:
   • Cancer Total: {stats['Cancer_total']:.1f}%
   • Inflammation: {stats['INF']:.1f}%
   • Follicle: {stats['FOL']:.1f}%
   • Background: {stats['BKG']:.1f}%

Confidence: {confidence_map.mean():.3f}
"""
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Comprehensive visualization saved to: {output_path}")


def process_single_image(
    image_path: str,
    model: SkinSegmentationModel,
    output_dir: str = "results",
    visualize: bool = True,
    neighborhood_size: int = NEIGHBORHOOD_SIZE
) -> Dict[str, float]:
    """Process a single image and return statistics."""
    
    # Check if this is a whole slide image
    if is_whole_slide_image(image_path):
        print(f"🔬 Detected whole slide image format: {image_path}")
        
        # Get target magnification from model or default to 10x
        target_magnification = getattr(model, 'magnification', '10x')
        if target_magnification == "unknown":
            target_magnification = "10x"
            print(f"⚠️  Model magnification unknown, defaulting to {target_magnification}")
        
        # Process WSI with tiles using proper infrastructure
        stats = process_wsi_with_tiles(
            image_path, model, tile_size=224, target_magnification=target_magnification, 
            output_dir=output_dir, neighborhood_size=neighborhood_size
        )
        
        if not stats:
            return {}
        
        print(f"✅ Processed WSI: {image_path}")
        print(f"   📊 Segmentation mosaics created in: {output_dir}")
        print(f"   📊 Epidermis {stats['Epithelial_total']:.1f}%, Dermis {stats['Dermis_total']:.1f}%")
        print(f"   🔬 Detailed nuclei: EPI: {stats['epi_nuclei']}, PAP: {stats['pap_nuclei']}, RET: {stats['ret_nuclei']}")
        
        return stats
    
    else:
        # Handle regular images
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return {}
        
        # Predict
        pred_mask, confidence_map = model.predict(image)
        
        # Calculate statistics
        stats = calculate_tissue_stats(pred_mask)
        
        # Create visualization if requested
        if visualize:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_segmentation.png")
            
            create_comprehensive_visualization(
                image, pred_mask, confidence_map, stats, output_path
            )
        
        print(f"✅ Processed: {image_path}")
        print(f"   📊 Key stats: Epidermis {stats['Epithelial_total']:.1f}%, "
              f"Dermis {stats['Dermis_total']:.1f}%")
        
        return stats


def process_batch(
    input_dir: str,
    model: SkinSegmentationModel,
    output_dir: str = "results",
    visualize: bool = True,
    neighborhood_size: int = NEIGHBORHOOD_SIZE,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ndpi', '.svs', '.vms', '.vmu', '.scn', '.mrxs', '.bif']
) -> List[Dict[str, float]]:
    """Process all images in a directory."""
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return []
    
    print(f"🔍 Found {len(image_files)} images to process")
    
    # Process each image
    all_stats = []
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {image_path.name}...")
        stats = process_single_image(str(image_path), model, output_dir, visualize, neighborhood_size)
        if stats:
            stats['filename'] = image_path.name
            all_stats.append(stats)
    
    # Save batch summary
    if all_stats:
        save_batch_summary(all_stats, output_dir)
    
    return all_stats


def save_batch_summary(all_stats: List[Dict[str, float]], output_dir: str) -> None:
    """Save batch processing summary to CSV."""
    import csv
    
    csv_path = os.path.join(output_dir, "batch_summary.csv")
    
    if not all_stats:
        return
    
    # Get all unique keys
    all_keys = set()
    for stats in all_stats:
        all_keys.update(stats.keys())
    
    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"📊 Batch summary saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Skin Histopathology Segmentation Inference",
        epilog="""
Examples:
  # Use any model from HuggingFace JoshuaFreeman/skin_seg:
  python skin_seg_inference.py image.jpg --model_name efficientnet-b3_10x
  python skin_seg_inference.py image.jpg --model_name efficientnet-b5
  python skin_seg_inference.py image.jpg --model_name efficientnet-b7_10x
  python skin_seg_inference.py image.jpg --model_name gigapath
  
  # Use magnification-specific models:
  python skin_seg_inference.py image.jpg --model_name efficientnet-b3_10x --magnification 10x
  python skin_seg_inference.py image.jpg --model_name efficientnet-b5_1x --magnification 1x
  
  # Use local model file:
  python skin_seg_inference.py image.jpg --model_path ./efficientnet-b3_10x_unet_best.pt --backbone efficientnet-b3
  
  # Batch processing:
  python skin_seg_inference.py /path/to/images/ --batch --model_name efficientnet-b3_10x
  
  # WSI with custom neighborhood size:
  python skin_seg_inference.py slide.ndpi --model_name efficientnet-b3_10x --neighborhood_size 5

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", help="Input image file or directory for batch processing")
    
    # Model specification options
    parser.add_argument("--model_name", help="Model name from JoshuaFreeman/skin_seg (e.g., 'efficientnet-b3_10x', 'efficientnet-b5', 'efficientnet-b7_10x', 'gigapath')")

    parser.add_argument("--model_path", help="Local path to model weights")
    parser.add_argument("--backbone", help="Backbone architecture (auto-detected if not provided)")
    parser.add_argument("--magnification", help="Expected magnification of input images (e.g., '1x', '2x', '5x', '10x'). If not specified, will try to auto-detect from model name")
    
    # Processing options
    parser.add_argument("--output_dir", default="results", help="Output directory for results")
    parser.add_argument("--batch", action="store_true", help="Process all images in input directory")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization generation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")
    parser.add_argument("--list_models", action="store_true", help="List available models in HuggingFace repo and exit")
    parser.add_argument("--neighborhood_size", type=int, default=NEIGHBORHOOD_SIZE, help=f"Neighborhood size for adaptive thresholding (default: {NEIGHBORHOOD_SIZE})")
    
    args = parser.parse_args()
    
    # List models and exit if requested
    if args.list_models:
        try:
            from huggingface_hub import list_repo_files
            repo_files = list_repo_files("JoshuaFreeman/skin_seg")
            model_files = [f for f in repo_files if f.endswith('.pt')]
            print("📋 Available models in JoshuaFreeman/skin_seg:")
            for model in sorted(model_files):
                print(f"   • {model}")
            print(f"\n💡 Usage: python skin_seg_inference.py image.jpg --model_name efficientnet-b3_10x")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        return
    
    # Validate arguments
    if not args.model_path and not args.model_name:
        # Default to efficientnet-b3_10x if nothing specified
        args.model_name = "efficientnet-b3_10x"
        print("💡 No model specified, defaulting to efficientnet-b3_10x")
    
    # Initialize model
    if args.model_name:
        print(f"🚀 Initializing model from HuggingFace: {args.model_name}")
        model = SkinSegmentationModel(
            model_name=args.model_name, 
            backbone=args.backbone, 
            device=args.device,
            requested_magnification=args.magnification
        )
    else:
        print(f"🚀 Initializing model from local file: {args.model_path}")
        model = SkinSegmentationModel(
            model_path=args.model_path, 
            backbone=args.backbone, 
            device=args.device,
            requested_magnification=args.magnification
        )
    
    # Show magnification warning if specified
    if args.magnification:
        print(f"⚠️  Input magnification specified as: {args.magnification}")
        if model.magnification != "unknown" and model.magnification != args.magnification:
            print(f"⚠️  WARNING: Model was trained on {model.magnification} data, but you specified {args.magnification}")
            print(f"   Performance may be degraded when using different magnifications!")
    elif model.magnification != "unknown":
        print(f"ℹ️  Model trained on {model.magnification} data. For best results, use matching magnification.")
    
    # Process input
    if args.batch or os.path.isdir(args.input):
        print(f"🔄 Batch processing directory: {args.input}")
        all_stats = process_batch(
            args.input, model, args.output_dir, 
            visualize=not args.no_visualize,
            neighborhood_size=args.neighborhood_size
        )
        print(f"\n🎉 Batch processing complete! Processed {len(all_stats)} images.")
        
    else:
        print(f"🔄 Processing single image: {args.input}")
        stats = process_single_image(
            args.input, model, args.output_dir,
            visualize=not args.no_visualize,
            neighborhood_size=args.neighborhood_size
        )
        print(f"\n🎉 Processing complete!")


if __name__ == "__main__":
    main()