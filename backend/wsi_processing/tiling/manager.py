import numpy as np
import cv2
import openslide
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from PIL import Image
import logging

from ..nuclei.detector import NucleiDetector


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
                 nuclei_count: int = 0, area_mm2: float = 0.0):
        self.tile_info = tile_info
        self.image = image
        self.nuclei_count = nuclei_count
        self.area_mm2 = area_mm2


class TilingManager:
    """Manages tiling and processing of whole slide images."""
    
    def __init__(self, 
                 tile_size: int = 128,
                 min_coverage_fraction: float = 0.5,
                 nuclei_detector: Optional[NucleiDetector] = None):
        """
        Args:
            tile_size: Size of tiles in pixels (at level 0)
            min_coverage_fraction: Minimum fraction of tile that must be inside ROI
            nuclei_detector: NucleiDetector instance for counting nuclei
        """
        self.tile_size = tile_size
        self.min_coverage_fraction = min_coverage_fraction
        self.nuclei_detector = nuclei_detector or NucleiDetector()
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
            "tiles_with_tissue": sum(1 for r in results if r.image is not None)
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
        
        # Detect nuclei
        gray_tile = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
        nuclei_count, nuclei_mask = self.nuclei_detector.detect_nuclei(tile_img, gray_tile)
        
        # Overlay nuclei
        tile_img[nuclei_mask == 255] = (0, 255, 0)  # Green for nuclei
        
        # Calculate area
        area_mm2 = inside_count * scale_factor * scale_factor * mpp_x * mpp_y * 1e-6
        
        return TileResult(tile_info, tile_img, nuclei_count, area_mm2)
    
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