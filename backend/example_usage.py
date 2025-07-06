#!/usr/bin/env python3
"""
Example usage of the WSI processing framework.
This script demonstrates how to use the refactored code.
"""

import os
import sys
import logging

# Add the wsi_processing module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wsi_processing'))

from wsi_processing import (
    WSIProcessor, 
    ProcessingConfig, 
    ConfigManager,
    GrabCutBackgroundMask,
    InverseBackgroundMask,
    ContourBasedMask
)


def example_single_file_processing():
    """Example: Process a single WSI file with custom configuration."""
    
    # Create custom configuration
    config = ProcessingConfig(
        input_dir="~/Downloads/ndpi_files",
        output_dir="~/Downloads/example_output"
    )
    
    # Customize nuclei detection parameters
    config.nuclei_detection.threshold_param = 15
    config.nuclei_detection.rpb_threshold_percentile = 45.0
    
    # Customize tiling parameters
    config.tiling.tile_size = 256
    config.tiling.min_coverage_fraction = 0.3
    
    # Create processor
    processor = WSIProcessor(
        contour_level=config.tiling.contour_level,
        tile_size=config.tiling.tile_size,
        min_tile_coverage=config.tiling.min_coverage_fraction
    )
    
    # Process a file (example path)
    slide_path = "~/Downloads/ndpi_files/example.ndpi"
    output_dir = "~/Downloads/example_output/example"
    
    if os.path.exists(os.path.expanduser(slide_path)):
        result = processor.process_slide(slide_path, output_dir)
        print(f"Processing completed. Total nuclei: {result['total_nuclei_count']}")
    else:
        print(f"File not found: {slide_path}")


def example_custom_masking():
    """Example: Create custom masking strategies."""
    
    # Create a custom masking pipeline
    background_mask = GrabCutBackgroundMask(iterations=10, rect_margin=0.1)
    tissue_mask = InverseBackgroundMask(background_mask)
    contour_mask = ContourBasedMask(
        base_mask_strategy=tissue_mask,
        kernel_size=(5, 5),
        top_n_contours=3,
        min_contour_area=5000
    )
    
    # Use with processor
    processor = WSIProcessor()
    
    # You can pass custom mask strategies to the processor
    # (This would require modifying the processor to accept mask strategies)
    print("Custom masking strategies created:")
    print(f"- {background_mask.get_name()}: {background_mask.get_parameters()}")
    print(f"- {tissue_mask.get_name()}")
    print(f"- {contour_mask.get_name()}: {contour_mask.get_parameters()}")


def example_batch_processing():
    """Example: Process multiple files in a directory."""
    
    from wsi_processing.core.processor import BatchProcessor
    
    # Create configuration
    config = ProcessingConfig(
        input_dir="~/Downloads/ndpi_files",
        output_dir="~/Downloads/batch_output",
        file_pattern="*.ndpi"
    )
    
    # Create batch processor
    batch_processor = BatchProcessor()
    
    # Process all files
    input_dir = os.path.expanduser(config.input_dir)
    output_dir = os.path.expanduser(config.output_dir)
    
    if os.path.exists(input_dir):
        results = batch_processor.process_directory(
            input_dir, output_dir, config.file_pattern
        )
        
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"Batch processing completed:")
        print(f"- Successful: {len(successful)} files")
        print(f"- Failed: {len(failed)} files")
        
        if successful:
            total_nuclei = sum(r['total_nuclei_count'] for r in successful)
            total_area = sum(r['total_non_background_area_mm2'] for r in successful)
            print(f"- Total nuclei counted: {total_nuclei}")
            print(f"- Total area analyzed: {total_area:.2f} mm²")
    else:
        print(f"Input directory not found: {input_dir}")


def example_configuration_management():
    """Example: Configuration file management."""
    
    # Create and save configuration
    config = ProcessingConfig()
    
    # Modify some parameters
    config.tiling.tile_size = 512
    config.nuclei_detection.threshold_param = 20
    config.masking.top_n_contours = 10
    
    # Save to file
    config_path = "example_config.yaml"
    ConfigManager.save_to_file(config, config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load from file
    loaded_config = ConfigManager.load_from_file(config_path)
    print(f"Configuration loaded. Tile size: {loaded_config.tiling.tile_size}")
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)


def main():
    """Run examples."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("WSI Processing Framework Examples")
    print("=" * 40)
    
    print("\n1. Custom Masking Strategies:")
    example_custom_masking()
    
    print("\n2. Configuration Management:")
    example_configuration_management()
    
    print("\n3. Single File Processing:")
    print("   (Requires actual NDPI file)")
    # example_single_file_processing()  # Uncomment if you have files
    
    print("\n4. Batch Processing:")
    print("   (Requires directory with NDPI files)")
    # example_batch_processing()  # Uncomment if you have files
    
    print("\nTo run actual processing, uncomment the relevant examples")
    print("and ensure you have NDPI files in ~/Downloads/ndpi_files/")


if __name__ == "__main__":
    main()