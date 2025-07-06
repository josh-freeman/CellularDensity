import numpy as np
import cv2
from .base import MaskStrategy


class GrabCutBackgroundMask(MaskStrategy):
    """
    Background masking using GrabCut algorithm.
    Returns a mask where background is 255 and foreground is 0.
    """
    
    def __init__(self, iterations: int = 5, rect_margin: float = 0.05):
        """
        Args:
            iterations: Number of GrabCut iterations
            rect_margin: Margin from image edges for initial rectangle (0-1)
        """
        self.iterations = iterations
        self.rect_margin = rect_margin
        
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate background mask using GrabCut algorithm."""
        # Create an initial mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Define a rectangle that likely contains the foreground
        height, width = image.shape[:2]
        margin_w = int(self.rect_margin * width)
        margin_h = int(self.rect_margin * height)
        rect = (margin_w, margin_h, width - 2*margin_w, height - 2*margin_h)
        
        # Allocate memory for models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Ensure the image is in 3-channel format
        if len(image.shape) == 2 or image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 
                    self.iterations, cv2.GC_INIT_WITH_RECT)
        
        # Extract background: 0 and 2 are background pixels
        background_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 255, 0).astype('uint8')
        
        return background_mask
    
    def get_name(self) -> str:
        return "GrabCut Background"
    
    def get_parameters(self) -> dict:
        return {
            "iterations": self.iterations,
            "rect_margin": self.rect_margin
        }


class InverseBackgroundMask(MaskStrategy):
    """
    Creates a tissue mask by inverting a background mask.
    Returns a mask where tissue is 255 and background is 0.
    """
    
    def __init__(self, background_strategy: MaskStrategy):
        self.background_strategy = background_strategy
        
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate tissue mask by inverting background mask."""
        background_mask = self.background_strategy.generate_mask(image)
        return cv2.bitwise_not(background_mask)
    
    def get_name(self) -> str:
        return f"Inverse of {self.background_strategy.get_name()}"
    
    def get_parameters(self) -> dict:
        return {
            "background_strategy": self.background_strategy.get_name(),
            "background_parameters": self.background_strategy.get_parameters()
        }


class ContourBasedMask(MaskStrategy):
    """
    Creates masks based on contour detection and filtering.
    """
    
    def __init__(self, base_mask_strategy: MaskStrategy, 
                 kernel_size: tuple = (3, 3),
                 top_n_contours: int = 5,
                 min_contour_area: float = 0.0):
        """
        Args:
            base_mask_strategy: Strategy to generate initial mask
            kernel_size: Kernel for morphological operations
            top_n_contours: Keep only the N largest contours
            min_contour_area: Minimum contour area to keep (in pixels)
        """
        self.base_mask_strategy = base_mask_strategy
        self.kernel_size = kernel_size
        self.top_n_contours = top_n_contours
        self.min_contour_area = min_contour_area
        
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate mask based on contour detection."""
        # Get base mask
        base_mask = self.base_mask_strategy.generate_mask(image)
        
        # Apply morphological operations to smooth
        kernel = np.ones(self.kernel_size, np.uint8)
        smoothed_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        if self.min_contour_area > 0:
            contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        # Sort by area and keep top N
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.top_n_contours]
        
        # Create new mask with filtered contours
        result_mask = np.zeros_like(base_mask)
        cv2.drawContours(result_mask, contours, -1, 255, thickness=cv2.FILLED)
        
        return result_mask
    
    def get_name(self) -> str:
        return f"Contour-based (top {self.top_n_contours})"
    
    def get_parameters(self) -> dict:
        return {
            "base_strategy": self.base_mask_strategy.get_name(),
            "kernel_size": self.kernel_size,
            "top_n_contours": self.top_n_contours,
            "min_contour_area": self.min_contour_area
        }