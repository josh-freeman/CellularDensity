"""
WSI Processing Package

A modular framework for processing whole slide images (WSI) with:
- Flexible masking strategies
- Nuclei detection and counting
- Tiling and parallel processing
- Configurable parameters
- Multiple export formats
"""

from .core.processor import WSIProcessor, BatchProcessor
from .config.settings import ProcessingConfig, ConfigManager
from .nuclei.detector import NucleiDetector
from .tiling.manager import TilingManager
from .masks.base import MaskStrategy, RelevancyMask
from .masks.background import GrabCutBackgroundMask, InverseBackgroundMask, ContourBasedMask

__version__ = "1.0.0"
__author__ = "WSI Processing Team"

__all__ = [
    "WSIProcessor",
    "BatchProcessor", 
    "ProcessingConfig",
    "ConfigManager",
    "NucleiDetector",
    "TilingManager",
    "MaskStrategy",
    "RelevancyMask",
    "GrabCutBackgroundMask",
    "InverseBackgroundMask", 
    "ContourBasedMask"
]