# WSI Processing Framework

A modular framework for processing whole slide images (WSI) with nuclei detection and counting.

## Features

- **Flexible Masking**: Multiple strategies for identifying regions of interest
- **Nuclei Detection**: Sophisticated nuclei counting using adaptive thresholding and R+B channel filtering
- **Tiling System**: Efficient parallel processing of large WSI files
- **Configurable Parameters**: Easy configuration management via YAML/JSON
- **Multiple Export Formats**: CSV, JSON, and text reports
- **Extensible Architecture**: Easy to add new masking strategies or analysis methods

## Architecture

```
wsi_processing/
├── config/          # Configuration management
├── core/           # Main processing orchestration
├── masks/          # Masking strategies
├── nuclei/         # Nuclei detection algorithms
├── tiling/         # Tile management and processing
├── utils/          # Utilities for export and visualization
└── main.py         # Main entry point script
```

## Quick Start

### Basic Usage

```bash
# Process all .ndpi files in a directory
python main.py --input-dir ~/Downloads/ndpi_files --output-dir ~/Downloads/results

# Process a single file
python main.py --input-file ~/Downloads/sample.ndpi --output-dir ~/Downloads/results

# Use custom configuration
python main.py --config config/default.yaml --input-dir ~/Downloads/ndpi_files
```

### Configuration

Create a configuration file (YAML or JSON):

```yaml
# config.yaml
input_dir: "~/Downloads/ndpi_files"
output_dir: "~/Downloads/results"
file_pattern: "*.ndpi"

nuclei_detection:
  threshold_param: 13
  kernel_size: [3, 3]
  
tiling:
  tile_size: 128
  contour_level: 6
```

### Programmatic Usage

```python
from wsi_processing import WSIProcessor, ProcessingConfig

# Create configuration
config = ProcessingConfig(
    input_dir="~/Downloads/ndpi_files",
    output_dir="~/Downloads/results"
)

# Create processor
processor = WSIProcessor()

# Process a slide
result = processor.process_slide("sample.ndpi", "output_dir")
```

## Extensibility

### Adding New Masking Strategies

```python
from wsi_processing.masks.base import MaskStrategy
import numpy as np

class CustomMaskStrategy(MaskStrategy):
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        # Your custom masking logic here
        return mask
    
    def get_name(self) -> str:
        return "Custom Mask"
```

### Adding New Analysis Methods

The framework is designed to be easily extensible. You can:

1. **Add new masking strategies** by inheriting from `MaskStrategy`
2. **Modify nuclei detection** by customizing the `NucleiDetector` class
3. **Add new export formats** by extending the `utils.export` module
4. **Implement new visualization** by extending the `utils.visualization` module

## Command Line Options

```bash
python main.py --help
```

Key options:
- `--input-dir`: Directory containing WSI files
- `--output-dir`: Directory for results
- `--input-file`: Single file to process
- `--config`: Configuration file path
- `--tile-size`: Tile size in pixels
- `--contour-level`: Pyramid level for contour detection
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Output Structure

```
output_dir/
├── slide1/
│   ├── mask_1_contour_1/
│   │   └── tile_mosaic.png
│   ├── mask_1_contour_2/
│   │   └── tile_mosaic.png
│   ├── overview.png
│   └── results.json
├── slide2/
│   └── ...
├── summary.csv
├── detailed.csv
├── results.json
├── analysis_report.txt
└── configuration.json
```

## Performance Considerations

- **Parallel Processing**: Tiles are processed in parallel for speed
- **Memory Management**: Large images are processed in chunks
- **Configurable Parameters**: Adjust tile size and coverage thresholds based on your hardware

## Original Code Migration

This framework replaces the original notebook-based workflow with:

1. **Modular Design**: Each component is separated into its own module
2. **Reusable Code**: Core functionality can be reused across different projects
3. **Better Testing**: Each module can be tested independently
4. **Configuration Management**: Easy parameter tuning without code changes
5. **Batch Processing**: Process multiple files with a single command

The original notebook functionality is preserved while providing a much more maintainable and extensible codebase.