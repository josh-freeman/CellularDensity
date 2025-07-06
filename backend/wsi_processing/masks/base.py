from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class MaskStrategy(ABC):
    """Abstract base class for different masking strategies."""
    
    @abstractmethod
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask from the input image.
        
        Args:
            image: Input image as numpy array (can be RGB or grayscale)
            
        Returns:
            Binary mask where 255 indicates regions of interest, 0 indicates regions to ignore
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this masking strategy."""
        pass
    
    def get_parameters(self) -> dict:
        """Return the parameters used by this masking strategy."""
        return {}


class RelevancyMask:
    """Container for a relevancy mask with metadata."""
    
    def __init__(self, mask: np.ndarray, name: str, strategy: MaskStrategy):
        self.mask = mask
        self.name = name
        self.strategy = strategy
        self._area_mm2: Optional[float] = None
        
    def get_area_mm2(self, mpp_x: float, mpp_y: float) -> float:
        """Calculate the area of the mask in mm²."""
        if self._area_mm2 is None:
            pixel_count = np.count_nonzero(self.mask)
            pixel_area_um2 = mpp_x * mpp_y
            self._area_mm2 = pixel_count * pixel_area_um2 * 1e-6
        return self._area_mm2
    
    def get_bounding_boxes(self) -> list:
        """Get bounding boxes of all connected components in the mask."""
        import cv2
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours]