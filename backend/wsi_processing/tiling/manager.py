import numpy as np
import cv2
import openslide
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from PIL import Image
import logging
from enum import Enum

from ..nuclei.detector import NucleiDetector


class TissueFilterMode(Enum):
    """Tissue filtering modes for cell counting."""
    NONE = 0          # No filtering, count all tissue
    EXCLUDE_STRUCTURAL = 1  # Exclude structural tissue (GLD+KER+HYP)
    DERMIS_HYPODERMIS_ONLY = 2  # Only count in dermis+hypodermis (RET+PAP+EPI)


class TileInfo:
    """Container for tile information."""
    def __init__(self, row: int, col: int, 
                 x0: int, y0: int, w0: int, h0: int,
                 x_scaled: int, y_scaled: int, w_scaled: int, h_scaled: int):
        self.row = row
        self.col = col
        self.x0 = x0  # Level 0 coordinates
        self.y0 = y0
        self.w0 = w0
        self.h0 = h0
        self.x_scaled = x_scaled  # Scaled coordinates
        self.y_scaled = y_scaled
        self.w_scaled = w_scaled
        self.h_scaled = h_scaled


class TileResult:
    """Container for tile processing results."""
    def __init__(self, tile_info: TileInfo, image: Optional[np.ndarray] = None,
                 nuclei_count: int = 0, area_mm2: float = 0.0, 
                 inference_tiles: Optional[List[Dict]] = None):
        self.tile_info = tile_info
        self.image = image
        self.nuclei_count = nuclei_count
        self.area_mm2 = area_mm2
        self.inference_tiles = inference_tiles or []


class TilingManager:
    """Manages tiling and processing of whole slide images."""
    
    def __init__(self, 
                 tile_size: int = 128,
                 min_coverage_fraction: float = 0.5,
                 nuclei_detector: Optional[NucleiDetector] = None,
                 inference_tile_size: int = 256,
                 segmentation_model = None,
                 tissue_filter_mode: TissueFilterMode = TissueFilterMode.NONE):
        """
        Args:
            tile_size: Size of tiles in pixels (at level 0)
            min_coverage_fraction: Minimum fraction of tile that must be inside ROI
            nuclei_detector: NucleiDetector instance for counting nuclei
            inference_tile_size: Size of tiles for inference processing
            segmentation_model: Model for generating BKG segmentation labels
            tissue_filter_mode: Tissue filtering mode for cell counting
        """
        self.tile_size = tile_size
        self.min_coverage_fraction = min_coverage_fraction
        self.nuclei_detector = nuclei_detector or NucleiDetector()
        self.inference_tile_size = inference_tile_size
        self.segmentation_model = segmentation_model
        self.tissue_filter_mode = tissue_filter_mode
        self.logger = logging.getLogger(__name__)
        
    def tile_region(self, 
                   slide: openslide.OpenSlide,
                   contour: np.ndarray,
                   contour_level: int,
                   tissue_mask: np.ndarray,
                   background_mask: np.ndarray,
                   output_dir: Optional[str] = None) -> Dict[str, any]:
        """
        Tile a region defined by a contour and process each tile.
        
        Args:
            slide: OpenSlide object
            contour: Contour points defining the region
            contour_level: Level at which contour was detected
            tissue_mask: Binary mask of tissue regions (at contour_level)
            background_mask: Binary mask of background (at contour_level)
            output_dir: Optional directory to save tile images
            
        Returns:
            Dictionary with results including total nuclei count and area
        """
        # Get scale factors
        scale_factor = slide.level_downsamples[contour_level]
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0.0))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0.0))
        
        # Get bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create contour-specific mask
        contour_mask = np.zeros_like(tissue_mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        
        # Convert to level 0 coordinates
        x0 = int(x * scale_factor)
        y0 = int(y * scale_factor)
        w0 = int(w * scale_factor)
        h0 = int(h * scale_factor)
        
        # Generate tile information
        tile_infos = self._generate_tile_infos(x0, y0, w0, h0, scale_factor)
        
        # Process tiles in parallel
        results = self._process_tiles(
            slide, tile_infos, contour_mask, background_mask,
            scale_factor, mpp_x, mpp_y
        )
        
        # Aggregate results
        total_nuclei = sum(r.nuclei_count for r in results)
        total_area_mm2 = self._calculate_region_area_mm2(
            contour_mask[y:y+h, x:x+w], scale_factor, mpp_x, mpp_y
        )
        
        # Create mosaic if output directory provided
        if output_dir:
            self._save_results(results, x0, y0, w0, h0, output_dir)
            
        return {
            "total_nuclei_count": total_nuclei,
            "total_area_mm2": total_area_mm2,
            "nuclei_density_per_mm2": total_nuclei / total_area_mm2 if total_area_mm2 > 0 else 0,
            "tiles_processed": len(results),
            "tiles_with_tissue": sum(1 for r in results if r.image is not None),
            "inference_tiles_total": sum(len(r.inference_tiles) for r in results),
            "inference_tiles_valid": sum(sum(1 for inf in r.inference_tiles if inf['valid']) for r in results),
            "tissue_filter_mode": self.tissue_filter_mode.name,
            "tissue_filter_description": self._get_tissue_filter_description()
        }
    
    def _generate_tile_infos(self, x0: int, y0: int, w0: int, h0: int, 
                           scale_factor: float) -> List[TileInfo]:
        """Generate tile information for a bounding box."""
        n_tiles_x = math.ceil(w0 / self.tile_size)
        n_tiles_y = math.ceil(h0 / self.tile_size)
        
        tile_infos = []
        for row in range(n_tiles_y):
            for col in range(n_tiles_x):
                tile_x0 = x0 + col * self.tile_size
                tile_y0 = y0 + row * self.tile_size
                tile_w0 = min(self.tile_size, x0 + w0 - tile_x0)
                tile_h0 = min(self.tile_size, y0 + h0 - tile_y0)
                
                if tile_w0 <= 0 or tile_h0 <= 0:
                    continue
                    
                # Scaled coordinates
                x_scaled = int(tile_x0 / scale_factor)
                y_scaled = int(tile_y0 / scale_factor)
                w_scaled = int(tile_w0 / scale_factor)
                h_scaled = int(tile_h0 / scale_factor)
                
                tile_infos.append(TileInfo(
                    row, col, tile_x0, tile_y0, tile_w0, tile_h0,
                    x_scaled, y_scaled, w_scaled, h_scaled
                ))
                
        return tile_infos
    
    def _process_tiles(self, slide: openslide.OpenSlide, 
                      tile_infos: List[TileInfo],
                      contour_mask: np.ndarray,
                      background_mask: np.ndarray,
                      scale_factor: float,
                      mpp_x: float, mpp_y: float) -> List[TileResult]:
        """Process tiles in parallel."""
        results = []
        
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._process_single_tile,
                    slide, tile_info, contour_mask, background_mask,
                    scale_factor, mpp_x, mpp_y
                ): tile_info
                for tile_info in tile_infos
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing tile: {e}")
                    
        return results
    
    def _process_single_tile(self, slide: openslide.OpenSlide,
                           tile_info: TileInfo,
                           contour_mask: np.ndarray,
                           background_mask: np.ndarray,
                           scale_factor: float,
                           mpp_x: float, mpp_y: float) -> TileResult:
        """Process a single tile."""
        # Check coverage
        max_y = contour_mask.shape[0]
        max_x = contour_mask.shape[1]
        
        x_end = min(tile_info.x_scaled + tile_info.w_scaled, max_x)
        y_end = min(tile_info.y_scaled + tile_info.h_scaled, max_y)
        
        if (x_end <= 0 or y_end <= 0 or 
            tile_info.x_scaled >= max_x or tile_info.y_scaled >= max_y):
            return TileResult(tile_info)
        
        # Check tissue coverage
        tile_mask_slice = contour_mask[
            tile_info.y_scaled:y_end,
            tile_info.x_scaled:x_end
        ]
        inside_count = cv2.countNonZero(tile_mask_slice)
        slice_area = (y_end - tile_info.y_scaled) * (x_end - tile_info.x_scaled)
        coverage_fraction = inside_count / slice_area if slice_area > 0 else 0.0
        
        if coverage_fraction < self.min_coverage_fraction:
            return TileResult(tile_info)
        
        # Read tile image
        region = slide.read_region(
            (tile_info.x0, tile_info.y0), 0,
            (tile_info.w0, tile_info.h0)
        ).convert("RGB")
        tile_img = np.array(region)
        
        # Overlay background mask
        bkg_slice = background_mask[
            tile_info.y_scaled:y_end,
            tile_info.x_scaled:x_end
        ]
        if bkg_slice.size > 0:
            bkg_resized = cv2.resize(
                bkg_slice,
                (tile_info.w0, tile_info.h0),
                interpolation=cv2.INTER_NEAREST
            )
            tile_img[bkg_resized == 255] = (0, 0, 255)  # Blue for background
        
        # Subdivide tile for inference if needed
        inference_tiles = []
        total_nuclei = 0
        
        if self.segmentation_model is not None:
            # Generate BKG mask from segmentation
            segmentation_mask = self._generate_segmentation_mask(tile_img)
            bkg_mask = (segmentation_mask == 8).astype(np.uint8) * 255  # Class 8 is BKG
            
            # Create tissue filter mask
            tissue_filter_mask = self._create_tissue_filter_mask(segmentation_mask)
            
            # Subdivide tile for inference
            inference_tiles = self._subdivide_tile_for_inference(tile_img, bkg_mask, tissue_filter_mask)
            
            # Run cell counting on each inference tile
            for inf_tile in inference_tiles:
                if inf_tile['valid']:
                    gray_inf_tile = cv2.cvtColor(inf_tile['image'], cv2.COLOR_RGB2GRAY)
                    nuclei_count, nuclei_mask = self.nuclei_detector.detect_nuclei(inf_tile['image'], gray_inf_tile)
                    
                    # Apply tissue filtering to nuclei mask
                    if inf_tile['tissue_filter_mask'] is not None:
                        # Only count nuclei in allowed tissue areas
                        filtered_nuclei_mask = cv2.bitwise_and(nuclei_mask, inf_tile['tissue_filter_mask'])
                        # Recalculate nuclei count based on filtered mask
                        filtered_nuclei_count = len(np.unique(cv2.connectedComponents(filtered_nuclei_mask)[1])) - 1  # -1 to exclude background
                        inf_tile['nuclei_count'] = max(0, filtered_nuclei_count)
                        inf_tile['nuclei_mask'] = filtered_nuclei_mask
                    else:
                        inf_tile['nuclei_count'] = nuclei_count
                        inf_tile['nuclei_mask'] = nuclei_mask
                    
                    total_nuclei += inf_tile['nuclei_count']
                    
                    # Overlay nuclei on inference tile
                    inf_tile['image'][inf_tile['nuclei_mask'] == 255] = (0, 255, 0)  # Green for nuclei
                    
                    # Apply BKG background from segmentation
                    inf_tile_bkg = inf_tile['bkg_mask']
                    inf_tile['image'][inf_tile_bkg == 255] = (0, 0, 0)  # Black for BKG background
            
            # Update main tile with smooth BKG background
            tile_img[bkg_mask == 255] = (0, 0, 0)  # Black for BKG background
            
            # Apply tissue filtering overlay to main tile
            tile_img = self._apply_tissue_overlay(tile_img, segmentation_mask)
            
        else:
            # Fallback to original nuclei detection
            gray_tile = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
            nuclei_count, nuclei_mask = self.nuclei_detector.detect_nuclei(tile_img, gray_tile)
            total_nuclei = nuclei_count
            
            # Overlay nuclei
            tile_img[nuclei_mask == 255] = (0, 255, 0)  # Green for nuclei
            
            # Use existing background mask
            tile_img[bkg_resized == 255] = (0, 0, 255)  # Blue for background
        
        # Calculate area
        area_mm2 = inside_count * scale_factor * scale_factor * mpp_x * mpp_y * 1e-6
        
        return TileResult(tile_info, tile_img, total_nuclei, area_mm2, inference_tiles)
    
    def _calculate_region_area_mm2(self, mask_slice: np.ndarray,
                                 scale_factor: float,
                                 mpp_x: float, mpp_y: float) -> float:
        """Calculate area of a region in mm²."""
        pixel_count = np.count_nonzero(mask_slice)
        pixel_area_um2 = scale_factor * scale_factor * mpp_x * mpp_y
        return pixel_count * pixel_area_um2 * 1e-6
    
    def _save_results(self, results: List[TileResult],
                     x0: int, y0: int, w0: int, h0: int,
                     output_dir: str):
        """Save tile results as mosaic image."""
        n_tiles_x = math.ceil(w0 / self.tile_size)
        n_tiles_y = math.ceil(h0 / self.tile_size)
        
        # Create grid
        tile_grid = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]
        
        for result in results:
            if result.image is not None:
                tile_grid[result.tile_info.row][result.tile_info.col] = result.image
        
        # Create mosaic
        mosaic = self._create_mosaic(tile_grid)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "tile_mosaic.png")
        Image.fromarray(mosaic).save(output_path)
        self.logger.info(f"Saved mosaic to {output_path}")
        
        # Save inference tiles if they exist
        self._save_inference_tiles(results, output_dir)
    
    def _create_mosaic(self, tile_grid: List[List[Optional[np.ndarray]]]) -> np.ndarray:
        """Create mosaic from tile grid."""
        n_tiles_y = len(tile_grid)
        n_tiles_x = len(tile_grid[0]) if n_tiles_y > 0 else 0
        
        mosaic_h = n_tiles_y * self.tile_size
        mosaic_w = n_tiles_x * self.tile_size
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        
        for row in range(n_tiles_y):
            for col in range(n_tiles_x):
                tile_img = tile_grid[row][col]
                if tile_img is not None:
                    y0 = row * self.tile_size
                    x0 = col * self.tile_size
                    h, w = tile_img.shape[:2]
                    mosaic[y0:y0 + h, x0:x0 + w] = tile_img
                    
        return mosaic
    
    def _generate_segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate segmentation mask using the segmentation model."""
        if self.segmentation_model is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        try:
            # Convert to PIL Image for model inference
            pil_image = Image.fromarray(image)
            
            # Run segmentation inference
            result = self.segmentation_model.predict(pil_image)
            # Handle both single mask and tuple (mask, confidence) returns
            if isinstance(result, tuple):
                mask = result[0]  # Get mask from tuple
            else:
                mask = result
            
            # Ensure mask is the right size
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            return mask
        except Exception as e:
            self.logger.warning(f"Segmentation inference failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _subdivide_tile_for_inference(self, tile_image: np.ndarray, bkg_mask: np.ndarray, tissue_filter_mask: np.ndarray = None) -> List[Dict]:
        """Subdivide a tile into inference-sized tiles."""
        h, w = tile_image.shape[:2]
        inference_tiles = []
        
        # Calculate number of inference tiles needed
        tiles_x = max(1, w // self.inference_tile_size)
        tiles_y = max(1, h // self.inference_tile_size)
        
        # Calculate actual tile sizes (may be smaller than inference_tile_size)
        tile_w = w // tiles_x
        tile_h = h // tiles_y
        
        for row in range(tiles_y):
            for col in range(tiles_x):
                # Calculate tile boundaries
                x0 = col * tile_w
                y0 = row * tile_h
                x1 = min((col + 1) * tile_w, w)
                y1 = min((row + 1) * tile_h, h)
                
                # Extract tile
                tile_img = tile_image[y0:y1, x0:x1].copy()
                tile_bkg = bkg_mask[y0:y1, x0:x1].copy()
                
                # Extract tissue filter mask if provided
                tile_tissue_filter = None
                if tissue_filter_mask is not None:
                    tile_tissue_filter = tissue_filter_mask[y0:y1, x0:x1].copy()
                
                # Check if tile has enough tissue (not mostly background)
                tissue_pixels = np.sum(tile_bkg == 0)  # Non-background pixels
                total_pixels = tile_bkg.size
                tissue_fraction = tissue_pixels / total_pixels if total_pixels > 0 else 0
                
                # Additional check for tissue filter if applicable
                if tile_tissue_filter is not None:
                    allowed_tissue_pixels = np.sum(tile_tissue_filter == 255)
                    allowed_tissue_fraction = allowed_tissue_pixels / total_pixels if total_pixels > 0 else 0
                    # Use the more restrictive fraction
                    tissue_fraction = min(tissue_fraction, allowed_tissue_fraction)
                
                is_valid = tissue_fraction >= self.min_coverage_fraction
                
                inference_tiles.append({
                    'row': row,
                    'col': col,
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1,
                    'image': tile_img,
                    'bkg_mask': tile_bkg,
                    'tissue_filter_mask': tile_tissue_filter,
                    'tissue_fraction': tissue_fraction,
                    'valid': is_valid,
                    'nuclei_count': 0,
                    'nuclei_mask': None
                })
        
        return inference_tiles
    
    def _save_inference_tiles(self, results: List[TileResult], output_dir: str):
        """Save inference tiles as individual images and create detailed mosaics."""
        if not results or not any(r.inference_tiles for r in results):
            return
            
        inference_dir = os.path.join(output_dir, "inference_tiles")
        os.makedirs(inference_dir, exist_ok=True)
        
        for tile_idx, result in enumerate(results):
            if not result.inference_tiles:
                continue
                
            tile_dir = os.path.join(inference_dir, f"tile_{tile_idx}")
            os.makedirs(tile_dir, exist_ok=True)
            
            # Save individual inference tiles
            for inf_idx, inf_tile in enumerate(result.inference_tiles):
                if inf_tile['valid']:
                    tissue_info = ""
                    if self.tissue_filter_mode != TissueFilterMode.NONE:
                        tissue_info = f"_filtered"
                    filename = f"inf_tile_{inf_idx}_nuclei_{inf_tile['nuclei_count']}{tissue_info}.png"
                    filepath = os.path.join(tile_dir, filename)
                    Image.fromarray(inf_tile['image']).save(filepath)
            
            # Create inference tile mosaic for this main tile
            inf_mosaic = self._create_inference_mosaic(result.inference_tiles)
            if inf_mosaic is not None:
                mosaic_path = os.path.join(tile_dir, "inference_mosaic.png")
                Image.fromarray(inf_mosaic).save(mosaic_path)
        
        self.logger.info(f"Saved inference tiles to {inference_dir}")
    
    def _create_inference_mosaic(self, inference_tiles: List[Dict]) -> Optional[np.ndarray]:
        """Create mosaic from inference tiles."""
        if not inference_tiles:
            return None
            
        # Find grid dimensions
        max_row = max(inf['row'] for inf in inference_tiles) + 1
        max_col = max(inf['col'] for inf in inference_tiles) + 1
        
        # Get tile dimensions from first valid tile
        first_valid = next((inf for inf in inference_tiles if inf['valid']), None)
        if first_valid is None:
            return None
            
        tile_h, tile_w = first_valid['image'].shape[:2]
        
        # Create mosaic
        mosaic_h = max_row * tile_h
        mosaic_w = max_col * tile_w
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        
        for inf_tile in inference_tiles:
            if inf_tile['valid']:
                row, col = inf_tile['row'], inf_tile['col']
                y0 = row * tile_h
                x0 = col * tile_w
                
                img = inf_tile['image']
                h, w = img.shape[:2]
                mosaic[y0:y0 + h, x0:x0 + w] = img
        
        return mosaic
    
    def _create_tissue_filter_mask(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """Create tissue filter mask based on the selected filtering mode."""
        if self.tissue_filter_mode == TissueFilterMode.NONE:
            # No filtering - all tissue is valid (exclude only background)
            return (segmentation_mask != 8).astype(np.uint8) * 255
        
        elif self.tissue_filter_mode == TissueFilterMode.EXCLUDE_STRUCTURAL:
            # Exclude structural tissue: GLD(0) + KER(7) + HYP(3)
            structural_mask = (segmentation_mask == 0) | (segmentation_mask == 7) | (segmentation_mask == 3)
            exclude_mask = structural_mask | (segmentation_mask == 8)  # Also exclude background
            return (~exclude_mask).astype(np.uint8) * 255
        
        elif self.tissue_filter_mode == TissueFilterMode.DERMIS_HYPODERMIS_ONLY:
            # Include only dermis+hypodermis: RET(4) + PAP(5) + EPI(6)
            include_mask = (segmentation_mask == 4) | (segmentation_mask == 5) | (segmentation_mask == 6)
            return include_mask.astype(np.uint8) * 255
        
        else:
            # Default to no filtering
            return (segmentation_mask != 8).astype(np.uint8) * 255
    
    def _get_tissue_overlay_color(self) -> Tuple[int, int, int]:
        """Get overlay color for filtered tissue areas."""
        if self.tissue_filter_mode == TissueFilterMode.EXCLUDE_STRUCTURAL:
            return (128, 128, 128)  # Gray for excluded structural tissue
        elif self.tissue_filter_mode == TissueFilterMode.DERMIS_HYPODERMIS_ONLY:
            return (0, 0, 255)  # Blue for included dermis+hypodermis
        else:
            return (0, 0, 0)  # No overlay for no filtering
    
    def _apply_tissue_overlay(self, image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Apply tissue filtering overlay to image."""
        if self.tissue_filter_mode == TissueFilterMode.NONE:
            return image
        
        overlay_color = self._get_tissue_overlay_color()
        image_with_overlay = image.copy()
        
        if self.tissue_filter_mode == TissueFilterMode.EXCLUDE_STRUCTURAL:
            # Overlay gray on structural tissue (GLD+KER+HYP)
            structural_mask = (segmentation_mask == 0) | (segmentation_mask == 7) | (segmentation_mask == 3)
            image_with_overlay[structural_mask] = (
                image_with_overlay[structural_mask] * (1 - alpha) + 
                np.array(overlay_color) * alpha
            ).astype(np.uint8)
        
        elif self.tissue_filter_mode == TissueFilterMode.DERMIS_HYPODERMIS_ONLY:
            # Overlay blue on dermis+hypodermis (RET+PAP+EPI)
            dermis_hypodermis_mask = (segmentation_mask == 4) | (segmentation_mask == 5) | (segmentation_mask == 6)
            image_with_overlay[dermis_hypodermis_mask] = (
                image_with_overlay[dermis_hypodermis_mask] * (1 - alpha) + 
                np.array(overlay_color) * alpha
            ).astype(np.uint8)
        
        return image_with_overlay
    
    def _get_tissue_filter_description(self) -> str:
        """Get description of current tissue filtering mode."""
        if self.tissue_filter_mode == TissueFilterMode.NONE:
            return "No tissue filtering - counting all tissue areas"
        elif self.tissue_filter_mode == TissueFilterMode.EXCLUDE_STRUCTURAL:
            return "Excluding structural tissue (GLD+KER+HYP) from cell counting"
        elif self.tissue_filter_mode == TissueFilterMode.DERMIS_HYPODERMIS_ONLY:
            return "Counting only in dermis+hypodermis (RET+PAP+EPI) tissue"
        else:
            return "Unknown tissue filtering mode"