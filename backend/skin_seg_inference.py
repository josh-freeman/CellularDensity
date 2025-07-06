#!/usr/bin/env python3
"""
Skin Segmentation Inference Script
==================================

Complete inference script for skin histopathology segmentation models.
Supports batch processing, visualization, and quantitative analysis.
Auto-downloads models from HuggingFace repository: JoshuaFreeman/skin_seg

Key Features:
- EXACT same preprocessing and model architecture as training pipeline
- Auto-detects backbone architecture from model names  
- Downloads any model from HuggingFace JoshuaFreeman/skin_seg repository
- Supports GigaPath, DINOv2, and EfficientNet architectures
- Generates 3 key tissue masks: Epidermis, Dermis (RET+PAP), Structural (GLD+KER+HYP)
- Comprehensive 6-panel visualizations with statistics
- Individual binary mask export
- Batch processing support

Usage Examples:
    # List available models:
    python skin_seg_inference.py --list_models
    
    # Use any HuggingFace model:
    python skin_seg_inference.py image.jpg --model_name efficientnet-b3
    python skin_seg_inference.py image.jpg --model_name efficientnet-b5
    python skin_seg_inference.py image.jpg --model_name gigapath
    
    # Use local model:
    python skin_seg_inference.py image.jpg --model_path ./my_model.pt
    
    # Batch processing:
    python skin_seg_inference.py /path/to/images/ --batch --model_name efficientnet-b5
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
    
    def __init__(self, model_path: str = None, model_name: str = None, backbone: str = None, device: str = "auto"):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Local path to model weights (optional if model_name provided)
            model_name: HuggingFace model name (e.g. "efficientnet-b3", "gigapath", etc.)
            backbone: Backbone architecture (auto-detected from model_name if not provided)
            device: Device to run inference on ("auto", "cuda", or "cpu")
        """
        self.device = self._get_device(device)
        
        # Auto-detect backbone from model name/path if not provided
        if backbone is None:
            backbone = self._detect_backbone(model_path, model_name)
        
        self.backbone = backbone
        
        # Get model path (download from HF if needed)
        if model_path is None and model_name is not None:
            model_path = self._download_from_hf(model_name)
        elif model_path is None:
            raise ValueError("Either model_path or model_name must be provided")
            
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _detect_backbone(self, model_path: str = None, model_name: str = None) -> str:
        """Auto-detect backbone architecture from model name or path."""
        source = model_name or model_path or ""
        source = source.lower()
        
        # Check for specific backbones
        if "efficientnet-b3" in source or "efficientnet_b3" in source:
            return "efficientnet-b3"
        elif "efficientnet-b5" in source or "efficientnet_b5" in source:
            return "efficientnet-b5"
        elif "efficientnet-b7" in source or "efficientnet_b7" in source:
            return "efficientnet-b7"
        elif "resnet50" in source:
            return "resnet50"
        elif "resnet34" in source:
            return "resnet34"
        elif "gigapath" in source:
            return "gigapath_vitl"
        elif "dinov2" in source:
            if "base" in source:
                return "vit_base_patch14_dinov2"
            elif "large" in source:
                return "vit_large_patch14_dinov2"
            elif "small" in source:
                return "vit_small_patch14_dinov2"
            else:
                return "vit_base_patch14_dinov2"  # default
        else:
            # Default fallback
            print(f"⚠️  Could not auto-detect backbone from '{source}', using efficientnet-b3")
            return "efficientnet-b3"
    
    def _download_from_hf(self, model_name: str) -> str:
        """Download model from HuggingFace repository."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # List available files in the repository
            try:
                repo_files = list_repo_files("JoshuaFreeman/skin_seg")
                model_files = [f for f in repo_files if f.endswith('.pt')]
                print(f"📋 Available models in JoshuaFreeman/skin_seg: {model_files}")
            except Exception as e:
                print(f"⚠️  Could not list repo files: {e}")
                model_files = []
            
            # Try to find exact match first
            target_file = None
            for pattern in [f"{model_name}_unet_best.pt", f"{model_name}.pt", model_name]:
                if pattern in model_files:
                    target_file = pattern
                    break
            
            # If no exact match, try partial matching
            if target_file is None:
                for file in model_files:
                    if model_name.lower() in file.lower():
                        target_file = file
                        break
            
            # Fallback to common names
            if target_file is None:
                if "efficientnet-b3" in model_name or "b3" in model_name:
                    target_file = "efficientnet-b3_unet_best.pt"
                elif "efficientnet-b5" in model_name or "b5" in model_name:
                    target_file = "efficientnet-b5_unet_best.pt"
                elif "gigapath" in model_name.lower():
                    target_file = "gigapath_unet_best.pt"
                else:
                    target_file = model_files[0] if model_files else "efficientnet-b3_unet_best.pt"
            
            print(f"📥 Downloading {target_file} from JoshuaFreeman/skin_seg...")
            
            model_path = hf_hub_download(
                repo_id="JoshuaFreeman/skin_seg",
                filename=target_file,
                cache_dir="./models"
            )
            
            print(f"✅ Downloaded model: {model_path}")
            return model_path
            
        except ImportError:
            print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        except Exception as e:
            print(f"❌ Error downloading model '{model_name}': {e}")
            raise
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the segmentation model with EXACT same architecture as training."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture EXACTLY like training
        if self.backbone == "gigapath_vitl":
            # Use custom GigaPath implementation (same as training)
            try:
                import timm
                class GigaPathSeg(torch.nn.Module):
                    def __init__(self, n_classes=12, pretrained=False):
                        super().__init__()
                        self.backbone = timm.create_model(
                            "hf_hub:prov-gigapath/prov-gigapath",
                            pretrained=pretrained,
                            num_classes=0
                        )
                        
                        # Get patch size
                        _p = self.backbone.patch_embed.patch_size
                        self.patch_h, self.patch_w = (_p, _p) if isinstance(_p, int) else _p
                        C = self.backbone.embed_dim  # 1024
                        
                        self.decoder = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(C, 512, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(512, 256, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(256, 128, 2, stride=2), torch.nn.GELU(),
                            torch.nn.Conv2d(128, n_classes, 1)
                        )
                    
                    def forward(self, x):
                        B, _, H, W = x.shape
                        tokens = self.backbone.forward_features(x)  # B, N+1, C
                        tokens = tokens[:, 1:, :]  # drop CLS
                        
                        # Reshape sequence back to 2-D grid
                        h = H // self.patch_h
                        w = W // self.patch_w
                        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)  # B, C, h, w
                        
                        mask = self.decoder(feat)
                        return torch.nn.functional.interpolate(
                            mask, size=(H, W), mode="bilinear", align_corners=False
                        )
                
                model = GigaPathSeg(n_classes=12, pretrained=False)
                print(f"✅ Created custom GigaPath model")
                
            except Exception as e:
                print(f"❌ Error creating GigaPath model: {e}")
                raise
                
        elif 'dinov2' in self.backbone:
            # Use custom DINOv2 implementation (same as training)
            try:
                import timm
                class DINOv2Seg(torch.nn.Module):
                    def __init__(self, model_name="vit_base_patch14_dinov2", n_classes=12, pretrained=False):
                        super().__init__()
                        self.backbone = timm.create_model(
                            model_name,
                            pretrained=pretrained,
                            num_classes=0,
                            img_size=224
                        )
                        
                        self.patch_h = self.patch_w = self.backbone.patch_embed.patch_size[0]
                        C = self.backbone.embed_dim
                        
                        self.decoder = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(C, 512, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(512, 256, 2, stride=2), torch.nn.GELU(),
                            torch.nn.ConvTranspose2d(256, 128, 2, stride=2), torch.nn.GELU(),
                            torch.nn.Conv2d(128, n_classes, 1)
                        )
                    
                    def forward(self, x):
                        B, _, H, W = x.shape
                        tokens = self.backbone.forward_features(x)  # B, N+1, C
                        tokens = tokens[:, 1:, :]  # drop CLS
                        
                        h = H // self.patch_h
                        w = W // self.patch_w
                        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)  # B, C, h, w
                        
                        mask = self.decoder(feat)
                        return torch.nn.functional.interpolate(
                            mask, size=(H, W), mode="bilinear", align_corners=False
                        )
                
                model = DINOv2Seg(model_name=self.backbone, n_classes=12, pretrained=False)
                print(f"✅ Created custom DINOv2 model")
                
            except Exception as e:
                print(f"❌ Error creating DINOv2 model: {e}")
                raise
        else:
            # Use segmentation-models-pytorch for standard backbones
            model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights=None,  # We'll load our fine-tuned weights
                classes=12,
                activation=None
            )
            print(f"✅ Created SMP model with {self.backbone} backbone")
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Loaded {self.backbone} model from {model_path}")
        return model
    
    def _get_transform(self):
        """Get the preprocessing transform - EXACT same as training pipeline."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Use EXACT same preprocessing as training (no augmentation)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ])
    
    def predict(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation prediction on an image using EXACT same preprocessing as training.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Tuple of (prediction_mask, confidence_map)
        """
        # Convert to numpy array (uint8 0-255, same as training)
        img_array = np.array(image.convert('RGB'))
        h, w = img_array.shape[:2]
        
        # Handle image resizing to 224x224 EXACTLY like training
        patch_size = 224
        if h > patch_size or w > patch_size:
            # Select center patch (same as training inference)
            if h > patch_size:
                y = (h - patch_size) // 2
            else:
                y = 0
            if w > patch_size:
                x = (w - patch_size) // 2
            else:
                x = 0
            img_array = img_array[y:y+patch_size, x:x+patch_size]
        
        # If image is smaller, pad it (same as training inference)
        if img_array.shape[0] < patch_size or img_array.shape[1] < patch_size:
            pad_h = max(0, patch_size - img_array.shape[0])
            pad_w = max(0, patch_size - img_array.shape[1])
            img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
        
        # Apply EXACT same transform as training
        transformed = self.transform(image=img_array)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        
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




def main():
    parser = argparse.ArgumentParser(
        description="Skin Histopathology Segmentation Inference",
        epilog="""
Examples:
  # Use any model from HuggingFace JoshuaFreeman/skin_seg:
  python skin_seg_inference.py image.jpg --model_name efficientnet-b3
  python skin_seg_inference.py image.jpg --model_name efficientnet-b5
  python skin_seg_inference.py image.jpg --model_name gigapath
  
  # Use local model file:
  python skin_seg_inference.py image.jpg --model_path ./my_model.pt --backbone resnet50
  
  # Batch processing:
  python skin_seg_inference.py /path/to/images/ --batch --model_name efficientnet-b5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", help="Input image file or directory for batch processing")
    
    # Model specification options
    parser.add_argument("--model_name", help="Model name from JoshuaFreeman/skin_seg (e.g., 'efficientnet-b3', 'efficientnet-b5', 'gigapath')")
    parser.add_argument("--model_path", help="Local path to model weights")
    parser.add_argument("--backbone", help="Backbone architecture (auto-detected if not provided)")
    
    # Processing options
    parser.add_argument("--output_dir", default="results", help="Output directory for results")
    parser.add_argument("--batch", action="store_true", help="Process all images in input directory")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization generation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")
    parser.add_argument("--list_models", action="store_true", help="List available models in HuggingFace repo and exit")
    
    args = parser.parse_args()
    
    # List models and exit if requested
    if args.list_models:
        try:
            from huggingface_hub import list_repo_files
            repo_files = list_repo_files("JoshuaFreeman/skin_seg")
            model_files = [f for f in repo_files if f.endswith('.pt')]
            print("📋 Available models in JoshuaFreeman/skin_seg:")
            for model in sorted(model_files):
                print(f"   • {model}")
            print(f"\n💡 Usage: python skin_seg_inference.py image.jpg --model_name efficientnet-b3")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        return
    
    # Validate arguments
    if not args.model_path and not args.model_name:
        # Default to efficientnet-b3 if nothing specified
        args.model_name = "efficientnet-b3"
        print("💡 No model specified, defaulting to efficientnet-b3")
    
    # Initialize model
    if args.model_name:
        print(f"🚀 Initializing model from HuggingFace: {args.model_name}")
        model = SkinSegmentationModel(
            model_name=args.model_name, 
            backbone=args.backbone, 
            device=args.device
        )
    else:
        print(f"🚀 Initializing model from local file: {args.model_path}")
        model = SkinSegmentationModel(
            model_path=args.model_path, 
            backbone=args.backbone, 
            device=args.device
        )
    
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