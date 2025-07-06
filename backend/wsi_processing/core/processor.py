import openslide
import numpy as np
import cv2
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image

from ..masks.base import MaskStrategy, RelevancyMask
from ..masks.background import GrabCutBackgroundMask, InverseBackgroundMask, ContourBasedMask
from ..nuclei.detector import NucleiDetector
from ..tiling.manager import TilingManager


class WSIProcessor:
    """Main processor for whole slide images."""
    
    def __init__(self,
                 contour_level: int = 6,
                 tile_size: int = 128,
                 min_tile_coverage: float = 0.5,
                 nuclei_detector: Optional[NucleiDetector] = None,
                 tiling_manager: Optional[TilingManager] = None):
        """
        Args:
            contour_level: Pyramid level for contour detection
            tile_size: Size of tiles in pixels
            min_tile_coverage: Minimum fraction of tile that must be in ROI
            nuclei_detector: NucleiDetector instance
            tiling_manager: TilingManager instance
        """
        self.contour_level = contour_level
        self.tile_size = tile_size
        self.min_tile_coverage = min_tile_coverage
        
        self.nuclei_detector = nuclei_detector or NucleiDetector()
        self.tiling_manager = tiling_manager or TilingManager(
            tile_size=tile_size,
            min_coverage_fraction=min_tile_coverage,
            nuclei_detector=self.nuclei_detector
        )
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def process_slide(self, 
                     slide_path: str,
                     output_dir: str,
                     mask_strategies: Optional[List[MaskStrategy]] = None) -> Dict[str, any]:
        """
        Process a whole slide image.
        
        Args:
            slide_path: Path to the slide file
            output_dir: Directory to save outputs
            mask_strategies: List of masking strategies to apply
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing slide: {slide_path}")
        
        # Open slide
        slide = openslide.OpenSlide(slide_path)
        
        # Get default mask strategies if none provided
        if mask_strategies is None:
            mask_strategies = self._get_default_mask_strategies()
        
        # Read image at contour level
        level_dims = slide.level_dimensions[self.contour_level]
        level_image = slide.read_region((0, 0), self.contour_level, level_dims)
        level_array = np.array(level_image.convert("RGB"))
        
        # Generate masks
        masks = self._generate_masks(level_array, mask_strategies)
        
        # Process each mask
        results = []
        total_nuclei = 0
        total_area_mm2 = 0
        
        for i, mask in enumerate(masks):
            self.logger.info(f"\nProcessing mask {i+1}/{len(masks)}: {mask.name}")
            
            # Get contours from mask
            contours = self._get_contours_from_mask(mask.mask)
            self.logger.info(f"Found {len(contours)} contours")
            
            # Process each contour
            for j, contour in enumerate(contours):
                self.logger.info(f"Processing contour {j+1}/{len(contours)}")
                
                # Create output directory for this contour
                contour_dir = os.path.join(output_dir, f"mask_{i+1}_contour_{j+1}")
                os.makedirs(contour_dir, exist_ok=True)
                
                # Process tiles in this contour
                result = self.tiling_manager.tile_region(
                    slide=slide,
                    contour=contour,
                    contour_level=self.contour_level,
                    tissue_mask=mask.mask,
                    background_mask=self._get_background_mask(level_array),
                    output_dir=contour_dir
                )
                
                result["mask_name"] = mask.name
                result["contour_index"] = j
                results.append(result)
                
                total_nuclei += result["total_nuclei_count"]
                total_area_mm2 += result["total_area_mm2"]
        
        # Save overview image with all masks
        self._save_overview_image(level_array, masks, os.path.join(output_dir, "overview.png"))
        
        # Prepare final results
        final_results = {
            "slide_path": slide_path,
            "total_nuclei_count": total_nuclei,
            "total_non_background_area_mm2": total_area_mm2,
            "average_density_per_mm2": total_nuclei / total_area_mm2 if total_area_mm2 > 0 else 0,
            "masks_applied": len(masks),
            "contours_processed": len(results),
            "detailed_results": results,
            "parameters": {
                "contour_level": self.contour_level,
                "tile_size": self.tile_size,
                "min_tile_coverage": self.min_tile_coverage
            }
        }
        
        # Save results
        output_path = os.path.join(output_dir, "results.json")
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        self.logger.info(f"Results saved to {output_path}")
        
        slide.close()
        return final_results
    
    def _get_default_mask_strategies(self) -> List[MaskStrategy]:
        """Get default masking strategies."""
        # Background mask using GrabCut
        background_strategy = GrabCutBackgroundMask(iterations=5, rect_margin=0.05)
        
        # Tissue mask (inverse of background)
        tissue_strategy = InverseBackgroundMask(background_strategy)
        
        # Contour-based mask (top 5 largest tissue regions)
        contour_strategy = ContourBasedMask(
            base_mask_strategy=tissue_strategy,
            kernel_size=(3, 3),
            top_n_contours=5,
            min_contour_area=1000
        )
        
        return [contour_strategy]
    
    def _generate_masks(self, image: np.ndarray, 
                       strategies: List[MaskStrategy]) -> List[RelevancyMask]:
        """Generate masks using provided strategies."""
        masks = []
        
        for strategy in strategies:
            self.logger.info(f"Generating mask: {strategy.get_name()}")
            mask = strategy.generate_mask(image)
            relevancy_mask = RelevancyMask(mask, strategy.get_name(), strategy)
            masks.append(relevancy_mask)
            
        return masks
    
    def _get_contours_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract contours from a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours
    
    def _get_background_mask(self, image: np.ndarray) -> np.ndarray:
        """Get background mask for the image."""
        strategy = GrabCutBackgroundMask()
        return strategy.generate_mask(image)
    
    def _save_overview_image(self, image: np.ndarray, masks: List[RelevancyMask], 
                           output_path: str):
        """Save an overview image showing all masks."""
        overlay = image.copy()
        
        # Define colors for different masks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        # Draw contours for each mask
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            contours = self._get_contours_from_mask(mask.mask)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # Add label
            if contours:
                x, y = contours[0][0][0]
                cv2.putText(overlay, mask.name, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save
        Image.fromarray(overlay).save(output_path)
        self.logger.info(f"Overview image saved to {output_path}")


class BatchProcessor:
    """Process multiple WSI files in batch."""
    
    def __init__(self, wsi_processor: Optional[WSIProcessor] = None):
        self.wsi_processor = wsi_processor or WSIProcessor()
        self.logger = logging.getLogger(__name__)
        
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*.ndpi") -> List[Dict[str, any]]:
        """
        Process all matching files in a directory.
        
        Args:
            input_dir: Directory containing WSI files
            output_dir: Directory to save outputs
            file_pattern: File pattern to match
            
        Returns:
            List of results for each processed file
        """
        import glob
        
        # Find matching files
        pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(pattern)
        self.logger.info(f"Found {len(files)} files matching pattern {file_pattern}")
        
        results = []
        for i, file_path in enumerate(files):
            self.logger.info(f"\nProcessing file {i+1}/{len(files)}: {file_path}")
            
            # Create output directory for this file
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(output_dir, file_name)
            
            try:
                result = self.wsi_processor.process_slide(
                    file_path, file_output_dir
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    "slide_path": file_path,
                    "error": str(e)
                })
        
        # Save summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Batch summary saved to {summary_path}")
        
        return results