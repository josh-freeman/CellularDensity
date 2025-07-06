#!/usr/bin/env python3
"""
Skin Segmentation Inference Script
==================================

Complete inference script for EfficientNet skin histopathology segmentation models.
Supports batch processing, visualization, and quantitative analysis.

Models available at: https://huggingface.co/JoshuaFreeman/skin_seg

Usage:
    python skin_seg_inference.py image.jpg
    python skin_seg_inference.py --batch /path/to/images/
    python skin_seg_inference.py image.jpg --model efficientnet-b5 --visualize
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import segmentation_models_pytorch as smp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Color map for 12-class segmentation
COLOR_MAP = {
    0: (108, 0, 115),    # GLD - Gland
    1: (145, 1, 122),    # INF - Inflammation  
    2: (216, 47, 148),   # FOL - Follicle
    3: (254, 246, 242),  # HYP - Hypodermis
    4: (181, 9, 130),    # RET - Reticular
    5: (236, 85, 157),   # PAP - Papillary
    6: (73, 0, 106),     # EPI - Epidermis
    7: (248, 123, 168),  # KER - Keratin
    8: (0, 0, 0),        # BKG - Background
    9: (127, 255, 255),  # BCC - Basal Cell Carcinoma
    10: (127, 255, 142), # SCC - Squamous Cell Carcinoma
    11: (255, 127, 127), # IEC - Inflammatory/Epithelial Cells
}

CLASS_NAMES = {
    0: "GLD", 1: "INF", 2: "FOL", 3: "HYP", 4: "RET", 5: "PAP",
    6: "EPI", 7: "KER", 8: "BKG", 9: "BCC", 10: "SCC", 11: "IEC"
}

FULL_CLASS_NAMES = {
    0: "Gland", 1: "Inflammation", 2: "Follicle", 3: "Hypodermis", 
    4: "Reticular", 5: "Papillary", 6: "Epidermis", 7: "Keratin",
    8: "Background", 9: "Basal Cell Carcinoma", 10: "Squamous Cell Carcinoma", 
    11: "Inflammatory/Epithelial Cells"
}


class SkinSegmentationModel:
    """Wrapper class for skin segmentation models."""
    
    def __init__(self, model_path: str, backbone: str = "efficientnet-b3", device: str = "auto"):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Path to the trained model weights
            backbone: Backbone architecture (efficientnet-b3 or efficientnet-b5)
            device: Device to run inference on ("auto", "cuda", or "cpu")
        """
        self.backbone = backbone
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the segmentation model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture
        model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights=None,  # We'll load our fine-tuned weights
            classes=12,
            activation=None
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Loaded {self.backbone} model from {model_path}")
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get the preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation prediction on an image.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Tuple of (prediction_mask, confidence_map)
        """
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get prediction and confidence
            confidence, pred_indices = torch.max(probs, dim=1)
            
            pred_mask = pred_indices.squeeze(0).cpu().numpy()
            confidence_map = confidence.squeeze(0).cpu().numpy()
            
        return pred_mask, confidence_map


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB color image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        colored[mask == class_id] = color
    
    return colored


def calculate_tissue_stats(mask: np.ndarray) -> Dict[str, float]:
    """Calculate tissue class statistics."""
    total_pixels = mask.size
    stats = {}
    
    for class_id in range(12):
        count = np.sum(mask == class_id)
        percentage = (count / total_pixels) * 100
        stats[CLASS_NAMES[class_id]] = percentage
    
    # Calculate specialized groupings
    stats["Epithelial_total"] = stats["EPI"]
    stats["Dermis_total"] = stats["RET"] + stats["PAP"] 
    stats["Structural_total"] = stats["GLD"] + stats["KER"] + stats["HYP"]
    stats["Cancer_total"] = stats["BCC"] + stats["SCC"] + stats["IEC"]
    
    return stats


def create_overlay_visualization(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create specialized tissue overlay visualization with key masks."""
    # Convert to numpy for processing
    img_array = np.array(image.resize((mask.shape[1], mask.shape[0])))
    overlay = img_array.copy()
    
    # Create colored overlays with transparency
    alpha = 0.7
    
    # Epidermis (red) - Class 6
    epi_mask = mask == 6
    overlay[epi_mask] = overlay[epi_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    
    # Dermis (blue) - Reticular + Papillary (Classes 4, 5)
    dermis_mask = (mask == 4) | (mask == 5)
    overlay[dermis_mask] = overlay[dermis_mask] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    
    # Structural (green) - Gland + Keratin + Hypodermis (Classes 0, 7, 3)
    struct_mask = (mask == 0) | (mask == 7) | (mask == 3)
    overlay[struct_mask] = overlay[struct_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    
    return Image.fromarray(overlay.astype(np.uint8))


def create_individual_mask_visualization(mask: np.ndarray, original_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create individual binary masks for key tissue types."""
    masks = {}
    
    # Epidermis mask (Class 6)
    masks['epidermis'] = (mask == 6).astype(np.uint8) * 255
    
    # Dermis mask (Classes 4, 5: RET + PAP)
    masks['dermis'] = ((mask == 4) | (mask == 5)).astype(np.uint8) * 255
    
    # Structural mask (Classes 0, 7, 3: GLD + KER + HYP)
    masks['structural'] = ((mask == 0) | (mask == 7) | (mask == 3)).astype(np.uint8) * 255
    
    # Non-structural mask (everything except GLD + KER + HYP)
    masks['non_structural'] = (~((mask == 0) | (mask == 7) | (mask == 3))).astype(np.uint8) * 255
    
    return masks


def create_comprehensive_visualization(
    image: Image.Image,
    pred_mask: np.ndarray,
    confidence_map: np.ndarray,
    stats: Dict[str, float],
    output_path: str
) -> None:
    """Create comprehensive visualization with multiple panels."""
    
    # Resize image to match mask
    image_resized = image.resize((pred_mask.shape[1], pred_mask.shape[0]))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Skin Histopathology Segmentation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Segmentation Result
    colored_mask = mask_to_color(pred_mask)
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('12-Class Segmentation')
    axes[0, 1].axis('off')
    
    # 3. Confidence Map
    conf_plot = axes[0, 2].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('Prediction Confidence')
    axes[0, 2].axis('off')
    plt.colorbar(conf_plot, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. Specialized Overlay
    overlay_img = create_overlay_visualization(image, pred_mask)
    axes[1, 0].imshow(overlay_img)
    axes[1, 0].set_title('Key Tissue Masks\n(Red: Epidermis, Blue: Dermis, Green: Structural)')
    axes[1, 0].axis('off')
    
    # 5. Class Distribution
    class_percentages = [stats[name] for name in CLASS_NAMES.values()]
    colors = [np.array(COLOR_MAP[i]) / 255.0 for i in range(12)]
    
    bars = axes[1, 1].bar(CLASS_NAMES.values(), class_percentages, color=colors)
    axes[1, 1].set_title('Tissue Class Distribution')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, class_percentages):
        if pct > 1:  # Only label if >1%
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 6. Summary Statistics
    axes[1, 2].axis('off')
    summary_text = f"""
KEY TISSUE MASKS:

🔴 EPIDERMIS: {stats['EPI']:.1f}%
   (Class 6 - Epidermis)

🔵 DERMIS: {stats['Dermis_total']:.1f}%
   • Reticular: {stats['RET']:.1f}%
   • Papillary: {stats['PAP']:.1f}%

🟢 STRUCTURAL: {stats['Structural_total']:.1f}%
   • Gland: {stats['GLD']:.1f}%
   • Keratin: {stats['KER']:.1f}%
   • Hypodermis: {stats['HYP']:.1f}%

OTHER CLASSES:
   • Cancer Total: {stats['Cancer_total']:.1f}%
   • Inflammation: {stats['INF']:.1f}%
   • Follicle: {stats['FOL']:.1f}%
   • Background: {stats['BKG']:.1f}%

Confidence: {confidence_map.mean():.3f}
"""
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Comprehensive visualization saved to: {output_path}")


def process_single_image(
    image_path: str,
    model: SkinSegmentationModel,
    output_dir: str = "results",
    visualize: bool = True
) -> Dict[str, float]:
    """Process a single image and return statistics."""
    
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return {}
    
    # Predict
    pred_mask, confidence_map = model.predict(image)
    
    # Calculate statistics
    stats = calculate_tissue_stats(pred_mask)
    
    # Create visualization if requested
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_segmentation.png")
        
        create_comprehensive_visualization(
            image, pred_mask, confidence_map, stats, output_path
        )
    
    print(f"✅ Processed: {image_path}")
    print(f"   📊 Key stats: Epithelial {stats['Epithelial_total']:.1f}%, "
          f"Dermis {stats['Dermis_total']:.1f}%, Cancer {stats['Cancer_total']:.1f}%")
    
    return stats


def process_batch(
    input_dir: str,
    model: SkinSegmentationModel,
    output_dir: str = "results",
    visualize: bool = True,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
) -> List[Dict[str, float]]:
    """Process all images in a directory."""
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return []
    
    print(f"🔍 Found {len(image_files)} images to process")
    
    # Process each image
    all_stats = []
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {image_path.name}...")
        stats = process_single_image(str(image_path), model, output_dir, visualize)
        if stats:
            stats['filename'] = image_path.name
            all_stats.append(stats)
    
    # Save batch summary
    if all_stats:
        save_batch_summary(all_stats, output_dir)
    
    return all_stats


def save_batch_summary(all_stats: List[Dict[str, float]], output_dir: str) -> None:
    """Save batch processing summary to CSV."""
    import csv
    
    csv_path = os.path.join(output_dir, "batch_summary.csv")
    
    if not all_stats:
        return
    
    # Get all unique keys
    all_keys = set()
    for stats in all_stats:
        all_keys.update(stats.keys())
    
    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"📊 Batch summary saved to: {csv_path}")


def download_model_from_hf(model_name: str) -> str:
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id="JoshuaFreeman/skin_seg",
            filename=f"{model_name}_unet_best.pt",
            cache_dir="./models"
        )
        print(f"📥 Downloaded model from Hugging Face: {model_path}")
        return model_path
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Skin Histopathology Segmentation Inference")
    parser.add_argument("input", help="Input image file or directory for batch processing")
    parser.add_argument("--model", default="efficientnet-b3", 
                       choices=["efficientnet-b3", "efficientnet-b5"],
                       help="Model backbone to use")
    parser.add_argument("--model_path", help="Path to model weights (auto-downloads if not provided)")
    parser.add_argument("--output_dir", default="results", help="Output directory for results")
    parser.add_argument("--batch", action="store_true", help="Process all images in input directory")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization generation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Try local file first, then download
        local_path = f"{args.model}_unet_best.pt"
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = download_model_from_hf(args.model)
    
    # Initialize model
    print(f"🚀 Initializing {args.model} model...")
    model = SkinSegmentationModel(model_path, args.model, args.device)
    
    # Process input
    if args.batch or os.path.isdir(args.input):
        print(f"🔄 Batch processing directory: {args.input}")
        all_stats = process_batch(
            args.input, model, args.output_dir, 
            visualize=not args.no_visualize
        )
        print(f"\n🎉 Batch processing complete! Processed {len(all_stats)} images.")
        
    else:
        print(f"🔄 Processing single image: {args.input}")
        stats = process_single_image(
            args.input, model, args.output_dir,
            visualize=not args.no_visualize
        )
        print(f"\n🎉 Processing complete!")


if __name__ == "__main__":
    main()