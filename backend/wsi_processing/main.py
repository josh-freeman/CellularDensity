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

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import ConfigManager, ProcessingConfig
from core.processor import WSIProcessor, BatchProcessor
from nuclei.detector import NucleiDetector
from tiling.manager import TilingManager
from masks.background import GrabCutBackgroundMask, InverseBackgroundMask, ContourBasedMask
from utils.export import export_all_formats
from utils.visualization import create_summary_plot


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


def create_wsi_processor(config: ProcessingConfig) -> WSIProcessor:
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
        nuclei_detector=nuclei_detector
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


def process_single_file(file_path: str, config: ProcessingConfig) -> dict:
    """Process a single WSI file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing single file: {file_path}")
    
    # Create processor
    processor = create_wsi_processor(config)
    
    # Create output directory
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(config.output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    try:
        result = processor.process_slide(file_path, output_dir)
        logger.info(f"Successfully processed {file_path}")
        return result
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'slide_path': file_path,
            'error': str(e)
        }


def process_directory(config: ProcessingConfig) -> List[dict]:
    """Process all files in a directory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing directory: {config.input_dir}")
    
    # Create processor
    processor = create_wsi_processor(config)
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
        result = process_single_file(args.input_file, config)
        results = [result]
        
    elif config.input_dir:
        if not os.path.exists(config.input_dir):
            logger.error(f"Input directory does not exist: {config.input_dir}")
            sys.exit(1)
        
        # Process directory
        results = process_directory(config)
        
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