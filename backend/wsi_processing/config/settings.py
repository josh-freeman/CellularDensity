import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import yaml


@dataclass
class NucleiDetectionConfig:
    """Configuration for nuclei detection."""
    threshold_param: int = 13
    kernel_size: tuple = (3, 3)
    dilation_iterations: int = 2
    rpb_threshold_percentile: float = 50.0


@dataclass
class MaskingConfig:
    """Configuration for masking strategies."""
    grabcut_iterations: int = 5
    rect_margin: float = 0.05
    kernel_size: tuple = (3, 3)
    top_n_contours: int = 5
    min_contour_area: float = 1000.0


@dataclass
class TilingConfig:
    """Configuration for tiling."""
    tile_size: int = 128
    min_coverage_fraction: float = 0.5
    contour_level: int = 6


@dataclass
class ProcessingConfig:
    """Main processing configuration."""
    input_dir: str = "~/Downloads/ndpi_files"
    output_dir: str = "~/Downloads/ndpi_files_analysis"
    file_pattern: str = "*.ndpi"
    
    # Sub-configurations
    nuclei_detection: NucleiDetectionConfig = None
    masking: MaskingConfig = None
    tiling: TilingConfig = None
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.nuclei_detection is None:
            self.nuclei_detection = NucleiDetectionConfig()
        if self.masking is None:
            self.masking = MaskingConfig()
        if self.tiling is None:
            self.tiling = TilingConfig()
            
        # Expand paths
        self.input_dir = os.path.expanduser(self.input_dir)
        self.output_dir = os.path.expanduser(self.output_dir)


class ConfigManager:
    """Manages configuration loading and saving."""
    
    @staticmethod
    def load_from_file(config_path: str) -> ProcessingConfig:
        """Load configuration from file (JSON or YAML)."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return ConfigManager._dict_to_config(data)
    
    @staticmethod
    def save_to_file(config: ProcessingConfig, config_path: str):
        """Save configuration to file."""
        data = ConfigManager._config_to_dict(config)
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    @staticmethod
    def create_default_config() -> ProcessingConfig:
        """Create default configuration."""
        return ProcessingConfig()
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> ProcessingConfig:
        """Convert dictionary to configuration object."""
        # Extract sub-configurations
        nuclei_data = data.get('nuclei_detection', {})
        masking_data = data.get('masking', {})
        tiling_data = data.get('tiling', {})
        
        # Create sub-configurations
        nuclei_config = NucleiDetectionConfig(**nuclei_data)
        masking_config = MaskingConfig(**masking_data)
        tiling_config = TilingConfig(**tiling_data)
        
        # Create main configuration
        main_data = {k: v for k, v in data.items() 
                    if k not in ['nuclei_detection', 'masking', 'tiling']}
        
        return ProcessingConfig(
            nuclei_detection=nuclei_config,
            masking=masking_config,
            tiling=tiling_config,
            **main_data
        )
    
    @staticmethod
    def _config_to_dict(config: ProcessingConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {
            'input_dir': config.input_dir,
            'output_dir': config.output_dir,
            'file_pattern': config.file_pattern,
            'log_level': config.log_level,
            'nuclei_detection': {
                'threshold_param': config.nuclei_detection.threshold_param,
                'kernel_size': config.nuclei_detection.kernel_size,
                'dilation_iterations': config.nuclei_detection.dilation_iterations,
                'rpb_threshold_percentile': config.nuclei_detection.rpb_threshold_percentile
            },
            'masking': {
                'grabcut_iterations': config.masking.grabcut_iterations,
                'rect_margin': config.masking.rect_margin,
                'kernel_size': config.masking.kernel_size,
                'top_n_contours': config.masking.top_n_contours,
                'min_contour_area': config.masking.min_contour_area
            },
            'tiling': {
                'tile_size': config.tiling.tile_size,
                'min_coverage_fraction': config.tiling.min_coverage_fraction,
                'contour_level': config.tiling.contour_level
            }
        }
    
    @staticmethod
    def override_from_args(config: ProcessingConfig, args: Dict[str, Any]) -> ProcessingConfig:
        """Override configuration with command-line arguments."""
        # Create a copy
        new_config = ProcessingConfig(
            input_dir=args.get('input_dir', config.input_dir),
            output_dir=args.get('output_dir', config.output_dir),
            file_pattern=args.get('file_pattern', config.file_pattern),
            log_level=args.get('log_level', config.log_level),
            nuclei_detection=config.nuclei_detection,
            masking=config.masking,
            tiling=config.tiling
        )
        
        # Override specific parameters
        if 'tile_size' in args:
            new_config.tiling.tile_size = args['tile_size']
        if 'contour_level' in args:
            new_config.tiling.contour_level = args['contour_level']
        if 'min_coverage_fraction' in args:
            new_config.tiling.min_coverage_fraction = args['min_coverage_fraction']
        if 'top_n_contours' in args:
            new_config.masking.top_n_contours = args['top_n_contours']
        if 'threshold_param' in args:
            new_config.nuclei_detection.threshold_param = args['threshold_param']
            
        return new_config