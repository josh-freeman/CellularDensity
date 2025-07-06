import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional
from PIL import Image


def create_overlay(original_image: np.ndarray, 
                  mask: np.ndarray, 
                  overlay_color: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Create an overlay image with colored mask."""
    overlay = original_image.copy()
    overlay[mask == 255] = ((1 - alpha) * overlay[mask == 255] + alpha * overlay_color).astype(np.uint8)
    return overlay


def save_comparison_image(original: np.ndarray,
                         masks: List[np.ndarray],
                         mask_names: List[str],
                         output_path: str,
                         colors: Optional[List[tuple]] = None):
    """Save a comparison image showing original and all masks."""
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    n_images = len(masks) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    # Show original
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Show masks
    for i, (mask, name) in enumerate(zip(masks, mask_names)):
        overlay = create_overlay(original, mask, np.array(colors[i % len(colors)]))
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(name)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def create_density_heatmap(results: List[dict], 
                          output_path: str,
                          title: str = "Nuclei Density Heatmap"):
    """Create a heatmap showing nuclei density across different regions."""
    densities = [r.get('nuclei_density_per_mm2', 0) for r in results]
    names = [r.get('mask_name', f"Region {i}") for i, r in enumerate(results)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(range(len(densities)), densities, color='skyblue', alpha=0.7)
    ax.set_xlabel('Regions')
    ax.set_ylabel('Nuclei Density (per mm²)')
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{density:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def create_summary_plot(results: List[dict], output_path: str):
    """Create a summary plot with multiple metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    names = [r.get('mask_name', f"Region {i}") for i, r in enumerate(results)]
    densities = [r.get('nuclei_density_per_mm2', 0) for r in results]
    nuclei_counts = [r.get('total_nuclei_count', 0) for r in results]
    areas = [r.get('total_area_mm2', 0) for r in results]
    tiles_processed = [r.get('tiles_processed', 0) for r in results]
    
    # Density bar chart
    ax1.bar(range(len(densities)), densities, color='skyblue', alpha=0.7)
    ax1.set_title('Nuclei Density (per mm²)')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    # Total nuclei count
    ax2.bar(range(len(nuclei_counts)), nuclei_counts, color='lightcoral', alpha=0.7)
    ax2.set_title('Total Nuclei Count')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    
    # Area coverage
    ax3.bar(range(len(areas)), areas, color='lightgreen', alpha=0.7)
    ax3.set_title('Area Coverage (mm²)')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    
    # Tiles processed
    ax4.bar(range(len(tiles_processed)), tiles_processed, color='gold', alpha=0.7)
    ax4.set_title('Tiles Processed')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()