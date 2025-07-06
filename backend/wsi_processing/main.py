#!/usr/bin/env python3
"""
Main script for WSI processing.
Replaces the functionality of the original Jupyter notebook.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional
import numpy as np

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wsi_processing.config.settings import ConfigManager, ProcessingConfig
from wsi_processing.core.processor import WSIProcessor, BatchProcessor
from wsi_processing.nuclei.detector import NucleiDetector
from wsi_processing.tiling.manager import TilingManager, TissueFilterMode
from wsi_processing.masks.background import GrabCutBackgroundMask, InverseBackgroundMask, ContourBasedMask
from wsi_processing.utils.export import export_all_formats
from wsi_processing.utils.visualization import create_summary_plot


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('wsi_processing.log')
        ]
    )


def create_wsi_processor(config: ProcessingConfig, tissue_filter_mode: TissueFilterMode = TissueFilterMode.NONE, segmentation_model=None) -> WSIProcessor:
    """Create WSI processor from configuration."""
    
    # Create nuclei detector
    nuclei_detector = NucleiDetector(
        threshold_param=config.nuclei_detection.threshold_param,
        kernel_size=config.nuclei_detection.kernel_size,
        dilation_iterations=config.nuclei_detection.dilation_iterations,
        rpb_threshold_percentile=config.nuclei_detection.rpb_threshold_percentile
    )
    
    # Create tiling manager
    tiling_manager = TilingManager(
        tile_size=config.tiling.tile_size,
        min_coverage_fraction=config.tiling.min_coverage_fraction,
        nuclei_detector=nuclei_detector,
        tissue_filter_mode=tissue_filter_mode,
        segmentation_model=segmentation_model
    )
    
    # Create WSI processor
    processor = WSIProcessor(
        contour_level=config.tiling.contour_level,
        tile_size=config.tiling.tile_size,
        min_tile_coverage=config.tiling.min_coverage_fraction,
        nuclei_detector=nuclei_detector,
        tiling_manager=tiling_manager
    )
    
    return processor


def process_single_file(file_path: str, config: ProcessingConfig, tissue_filter_mode: TissueFilterMode = TissueFilterMode.NONE, segmentation_model=None) -> dict:
    """Process a single WSI file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing single file: {file_path}")
    
    # Create processor
    processor = create_wsi_processor(config, tissue_filter_mode, segmentation_model)
    
    # Create output directory
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(config.output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    try:
        result = processor.process_slide(file_path, output_dir)
        
        # Extract tissue patches for segmentation at correct resolution
        extract_and_segment_patches(file_path, output_dir, result)
        
        logger.info(f"Successfully processed {file_path}")
        return result
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'slide_path': file_path,
            'error': str(e)
        }

def extract_and_segment_patches(slide_path: str, output_dir: str, cell_results: dict):
    """Extract tissue patches and run segmentation analysis."""
    import openslide
    from PIL import Image
    import subprocess
    import json
    
    logger = logging.getLogger(__name__)
    logger.info("Extracting tissue patches for segmentation analysis...")
    
    try:
        # Open slide
        slide = openslide.OpenSlide(slide_path)
        
        # Determine correct level for 10x equivalent
        objective_power = float(slide.properties.get('openslide.objective-power', '40'))
        target_magnification = 10.0
        target_level = 0
        
        # Find level that gives 10x equivalent
        for level in range(slide.level_count):
            effective_mag = objective_power / slide.level_downsamples[level]
            if abs(effective_mag - target_magnification) < abs(objective_power / slide.level_downsamples[target_level] - target_magnification):
                target_level = level
        
        logger.info(f"Using level {target_level} for 10x equivalent magnification")
        
        # Create patches directory
        patches_dir = os.path.join(output_dir, "segmentation_patches")
        os.makedirs(patches_dir, exist_ok=True)
        
        # Extract patches from tissue regions
        level_dims = slide.level_dimensions[target_level]
        scale_factor = slide.level_downsamples[target_level]
        patch_size = 224
        patches_extracted = 0
        max_patches = 20
        
        # Sample patches from the tissue region
        # Use a grid sampling approach within the tissue bounds
        width, height = level_dims
        step_size = patch_size // 2  # 50% overlap
        
        patch_info = []
        
        for y in range(0, height - patch_size, step_size):
            for x in range(0, width - patch_size, step_size):
                if patches_extracted >= max_patches:
                    break
                
                # Convert to level 0 coordinates
                x_l0 = int(x * scale_factor)
                y_l0 = int(y * scale_factor)
                
                try:
                    # Extract patch
                    patch = slide.read_region((x_l0, y_l0), target_level, (patch_size, patch_size))
                    patch_rgb = patch.convert('RGB')
                    patch_array = np.array(patch_rgb)
                    
                    # Check tissue content
                    gray = np.mean(patch_array, axis=2)
                    white_ratio = np.sum(gray > 200) / (patch_size * patch_size)
                    
                    if white_ratio < 0.7:  # Good tissue content
                        patch_filename = f"patch_{patches_extracted:03d}_10x_224px.png"
                        patch_path = os.path.join(patches_dir, patch_filename)
                        patch_rgb.save(patch_path)
                        
                        patch_info.append({
                            'filename': patch_filename,
                            'coordinates': [x_l0, y_l0],
                            'level': target_level,
                            'white_ratio': white_ratio
                        })
                        
                        patches_extracted += 1
                        
                except Exception as e:
                    continue
            
            if patches_extracted >= max_patches:
                break
        
        slide.close()
        
        if patches_extracted == 0:
            logger.warning("No tissue patches extracted")
            return
        
        logger.info(f"Extracted {patches_extracted} tissue patches")
        
        # Save patch metadata
        metadata = {
            'source_slide': slide_path,
            'target_level': target_level,
            'patch_size': patch_size,
            'patches_extracted': patches_extracted,
            'patches': patch_info
        }
        
        metadata_path = os.path.join(patches_dir, "patch_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run segmentation on patches
        segmentation_dir = os.path.join(output_dir, "segmentation_results")
        
        # Find the skin segmentation script path
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        segmentation_script = os.path.join(script_dir, "skin_seg_inference.py")
        
        if os.path.exists(segmentation_script):
            logger.info("Running segmentation analysis on patches...")
            
            cmd = [
                "python", segmentation_script,
                patches_dir,
                "--batch",
                "--model_name", "efficientnet-b3", 
                "--output_dir", segmentation_dir
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("Segmentation analysis completed")
                
                # Create segmentation tile mosaic
                create_segmentation_mosaic(segmentation_dir, output_dir, patches_extracted)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Segmentation failed: {e}")
        else:
            logger.warning(f"Segmentation script not found at {segmentation_script}")
            
    except Exception as e:
        logger.error(f"Error in patch extraction and segmentation: {e}")

def create_segmentation_mosaic(segmentation_dir: str, output_dir: str, num_patches: int):
    """Create a tile mosaic of segmented patches."""
    import math
    from PIL import Image
    
    logger = logging.getLogger(__name__)
    
    try:
        # Find segmentation result images
        seg_files = [f for f in os.listdir(segmentation_dir) if f.endswith('_segmentation.png')]
        seg_files.sort()
        
        if not seg_files:
            logger.warning("No segmentation results found for mosaic")
            return
        
        # Calculate mosaic dimensions
        cols = math.ceil(math.sqrt(len(seg_files)))
        rows = math.ceil(len(seg_files) / cols)
        
        # Load first image to get dimensions
        first_img = Image.open(os.path.join(segmentation_dir, seg_files[0]))
        img_width, img_height = first_img.size
        
        # Create mosaic canvas
        mosaic_width = cols * img_width
        mosaic_height = rows * img_height
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color='white')
        
        # Place segmentation images in grid
        for i, seg_file in enumerate(seg_files):
            row = i // cols
            col = i % cols
            
            img = Image.open(os.path.join(segmentation_dir, seg_file))
            x = col * img_width
            y = row * img_height
            mosaic.paste(img, (x, y))
        
        # Save mosaic
        mosaic_path = os.path.join(output_dir, "segmentation_mosaic.png")
        mosaic.save(mosaic_path)
        
        logger.info(f"Segmentation mosaic saved: {mosaic_path}")
        logger.info(f"Mosaic contains {len(seg_files)} segmented patches in {rows}x{cols} grid")
        
    except Exception as e:
        logger.error(f"Error creating segmentation mosaic: {e}")


def process_directory(config: ProcessingConfig, tissue_filter_mode: TissueFilterMode = TissueFilterMode.NONE, segmentation_model=None) -> List[dict]:
    """Process all files in a directory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing directory: {config.input_dir}")
    
    # Create processor
    processor = create_wsi_processor(config, tissue_filter_mode, segmentation_model)
    batch_processor = BatchProcessor(processor)
    
    # Process all files
    results = batch_processor.process_directory(
        config.input_dir,
        config.output_dir,
        config.file_pattern
    )
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WSI Processing Pipeline")
    
    # Input/output options
    parser.add_argument('--input-dir', type=str, help='Input directory containing WSI files')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--input-file', type=str, help='Single input file to process')
    parser.add_argument('--file-pattern', type=str, default='*.ndpi', 
                       help='File pattern to match (default: *.ndpi)')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--save-config', type=str, help='Save configuration to file')
    
    # Processing parameters
    parser.add_argument('--tile-size', type=int, help='Tile size in pixels')
    parser.add_argument('--contour-level', type=int, help='Pyramid level for contour detection')
    parser.add_argument('--min-coverage-fraction', type=float, 
                       help='Minimum tile coverage fraction')
    parser.add_argument('--top-n-contours', type=int, help='Number of top contours to keep')
    parser.add_argument('--threshold-param', type=int, help='Nuclei detection threshold parameter')
    
    # Tissue filtering options
    parser.add_argument('--tissue-filter', type=str, 
                       choices=['none', 'exclude-structural', 'dermis-hypodermis-only'],
                       default='none',
                       help='Tissue filtering mode for cell counting')
    parser.add_argument('--segmentation-model', type=str,
                       help='Path to segmentation model or HuggingFace model name')
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Export options
    parser.add_argument('--export-formats', nargs='+', 
                       choices=['csv', 'json', 'report', 'plots'],
                       default=['csv', 'json', 'report'],
                       help='Export formats to generate')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = ConfigManager.load_from_file(args.config)
    else:
        logger.info("Using default configuration")
        config = ConfigManager.create_default_config()
    
    # Override with command line arguments
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    config = ConfigManager.override_from_args(config, args_dict)
    
    # Parse tissue filter mode
    tissue_filter_map = {
        'none': TissueFilterMode.NONE,
        'exclude-structural': TissueFilterMode.EXCLUDE_STRUCTURAL,
        'dermis-hypodermis-only': TissueFilterMode.DERMIS_HYPODERMIS_ONLY
    }
    tissue_filter_mode = tissue_filter_map[args.tissue_filter]
    
    # Load segmentation model if provided
    segmentation_model = None
    if args.segmentation_model:
        logger.info(f"Loading segmentation model: {args.segmentation_model}")
        try:
            # Import segmentation model class
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from skin_seg_inference import SkinSegmentationModel
            
            segmentation_model = SkinSegmentationModel(model_name=args.segmentation_model)
            logger.info("Segmentation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            logger.warning("Proceeding without segmentation model")
    
    # Save configuration if requested
    if args.save_config:
        logger.info(f"Saving configuration to {args.save_config}")
        ConfigManager.save_to_file(config, args.save_config)
    
    # Validate inputs
    if args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file does not exist: {args.input_file}")
            sys.exit(1)
        
        # Process single file
        result = process_single_file(args.input_file, config, tissue_filter_mode, segmentation_model)
        results = [result]
        
    elif config.input_dir:
        if not os.path.exists(config.input_dir):
            logger.error(f"Input directory does not exist: {config.input_dir}")
            sys.exit(1)
        
        # Process directory
        results = process_directory(config, tissue_filter_mode, segmentation_model)
        
    else:
        logger.error("Must specify either --input-file or --input-dir")
        sys.exit(1)
    
    # Export results
    if results:
        logger.info("Exporting results...")
        exported_files = export_all_formats(results, config.output_dir, config)
        
        # Create summary plots if requested
        if 'plots' in args.export_formats:
            successful_results = [r for r in results if 'error' not in r]
            if successful_results:
                # Flatten detailed results for plotting
                plot_data = []
                for result in successful_results:
                    detailed_results = result.get('detailed_results', [])
                    plot_data.extend(detailed_results)
                
                if plot_data:
                    plot_path = os.path.join(config.output_dir, "summary_plots.png")
                    create_summary_plot(plot_data, plot_path)
                    exported_files['plots'] = plot_path
        
        # Print summary
        successful_count = len([r for r in results if 'error' not in r])
        failed_count = len([r for r in results if 'error' in r])
        
        logger.info(f"Processing complete!")
        logger.info(f"  Successful: {successful_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Results exported to: {config.output_dir}")
        
        for format_name, file_path in exported_files.items():
            logger.info(f"  {format_name}: {file_path}")
    
    else:
        logger.warning("No results to export")


if __name__ == "__main__":
    main()