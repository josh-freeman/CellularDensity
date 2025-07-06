import numpy as np
import cv2
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor


class NucleiDetector:
    """Handles nuclei detection and counting in images."""
    
    def __init__(self, 
                 threshold_param: int = 13,
                 kernel_size: tuple = (3, 3),
                 dilation_iterations: int = 2,
                 rpb_threshold_percentile: float = 50.0):
        """
        Args:
            threshold_param: Parameter for adaptive thresholding
            kernel_size: Kernel size for morphological operations
            dilation_iterations: Number of dilation iterations
            rpb_threshold_percentile: Percentile threshold for R+B channel filtering
        """
        self.threshold_param = threshold_param
        self.kernel_size = kernel_size
        self.dilation_iterations = dilation_iterations
        self.rpb_threshold_percentile = rpb_threshold_percentile
        
    def detect_nuclei(self, image: np.ndarray, gray_image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Detect nuclei in the image and return count and mask.
        
        Args:
            image: RGB image as numpy array
            gray_image: Grayscale version of the image
            
        Returns:
            Tuple of (nuclei_count, nuclei_mask)
        """
        # Get initial nuclei mask
        initial_mask = self._get_initial_nuclei_mask(gray_image)
        
        # Dilate to connect nearby regions
        kernel = np.ones(self.kernel_size, np.uint8)
        dilated_mask = cv2.dilate(initial_mask, kernel, iterations=self.dilation_iterations)
        
        # Get connected components (zones)
        zones = self._get_connected_components(dilated_mask)
        
        # Calculate R+B threshold
        rpb_threshold = self._calculate_rpb_threshold(image, zones)
        
        # Filter zones based on R+B values
        filtered_mask, nuclei_count = self._filter_zones_by_rpb(image, zones, rpb_threshold)
        
        return nuclei_count, filtered_mask
    
    def _get_initial_nuclei_mask(self, gray_image: np.ndarray) -> np.ndarray:
        """Generate initial nuclei mask using adaptive thresholding."""
        thresh = cv2.adaptiveThreshold(
            gray_image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            self.threshold_param, 1
        )
        
        # Clean up small noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours and create filled mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nuclei_mask = np.zeros_like(thresh)
        cv2.drawContours(nuclei_mask, contours, -1, 255, thickness=cv2.FILLED)
        
        return nuclei_mask
    
    def _get_connected_components(self, mask: np.ndarray) -> List[np.ndarray]:
        """Get connected components from binary mask."""
        num_labels, labels = cv2.connectedComponents(mask)
        zones = []
        
        for label in range(1, num_labels):  # Skip background (label 0)
            coords = np.argwhere(labels == label)
            zones.append(coords)
            
        return zones
    
    def _calculate_rpb_threshold(self, image: np.ndarray, zones: List[np.ndarray]) -> int:
        """Calculate R+B channel threshold using Otsu's method."""
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image
            
        # Calculate histogram
        histogram = np.zeros((256, 3), dtype=int)
        
        for zone in zones:
            total_rgb = np.zeros(3, dtype=int)
            for x, y in zone:
                total_rgb += rgb_image[x, y]
            avg_rgb = total_rgb // len(zone)
            for i, value in enumerate(avg_rgb):
                histogram[value, i] += 1
        
        # Apply Otsu to get threshold percentile
        percentile_otsu = self._otsu_threshold(histogram)
        
        # Calculate actual threshold value
        rb_histogram = histogram[:, 0] + histogram[:, 2]
        cumulative_sum = np.cumsum(rb_histogram)
        total_pixels = cumulative_sum[-1]
        target_value = total_pixels * (percentile_otsu / 100.0)
        rb_threshold = np.searchsorted(cumulative_sum, target_value)
        
        return rb_threshold
    
    def _otsu_threshold(self, histogram: np.ndarray) -> float:
        """Calculate optimal threshold using Otsu's method."""
        rb_histogram = histogram[:, 0] + histogram[:, 2]
        total_pixels = np.sum(rb_histogram)
        
        sum_total = np.sum([i * rb_histogram[i] for i in range(256)])
        sum_background = 0
        weight_background = 0
        max_variance = 0
        threshold = 0
        
        for t in range(256):
            weight_background += rb_histogram[t]
            if weight_background == 0:
                continue
                
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break
                
            sum_background += t * rb_histogram[t]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            
            if variance_between > max_variance:
                max_variance = variance_between
                threshold = t
                
        return (threshold / 255) * 100
    
    def _filter_zones_by_rpb(self, image: np.ndarray, zones: List[np.ndarray], 
                            rpb_threshold: int) -> Tuple[np.ndarray, int]:
        """Filter zones based on R+B channel values."""
        filtered_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        valid_zones_count = 0
        
        def process_zone(zone):
            total_rpb = sum(int(image[x, y, 0]) + int(image[x, y, 2]) for x, y in zone)
            avg_rpb = total_rpb / len(zone)
            if avg_rpb <= rpb_threshold:
                return zone
            return None
        
        # Process zones in parallel
        with ThreadPoolExecutor() as executor:
            filtered_zones = list(executor.map(process_zone, zones))
        
        for zone in filtered_zones:
            if zone is not None:
                valid_zones_count += 1
                for x, y in zone:
                    filtered_mask[x, y] = 255
                    
        return filtered_mask, valid_zones_count