#!/usr/bin/env python3
"""
Precompute all 224x224 crops from the dataset and save them as individual files.
This eliminates the I/O bottleneck during training.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Color map from train_segmentation.py
COLOR2IDX = {
    (108,   0, 115): 0,  # GLD - Gland
    (145,   1, 122): 1,  # INF - Inflammation
    (216,  47, 148): 2,  # FOL - Follicle
    (254, 246, 242): 3,  # HYP - Hypodermis
    (181,   9, 130): 4,  # RET - Reticular
    (236,  85, 157): 5,  # PAP - Papillary
    ( 73,   0, 106): 6,  # EPI - Epidermis
    (248, 123, 168): 7,  # KER - Keratin
    (  0,   0,   0): 8,  # BKG - Background
    (127, 255, 255): 9,  # BCC - Basal Cell Carcinoma
    (127, 255, 142):10,  # SCC - Squamous Cell Carcinoma
    (255, 127, 127):11,  # IEC - Inflammatory/Epithelial Cells
}
IGNORE_LABEL = 255

def rgb_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB mask to index mask."""
    idx_mask = np.full(mask_rgb.shape[:2], IGNORE_LABEL, dtype=np.uint8)
    for rgb, idx in COLOR2IDX.items():
        matches = np.all(mask_rgb == rgb, axis=-1)
        idx_mask[matches] = idx
    return idx_mask

def extract_and_save_crops(root_dir, split_file, output_dir, img_ext=".tif", mask_ext=".png", 
                          crop_size=224, max_bg_ratio=0.9):
    """Extract all valid crops and save them as individual files."""
    root = Path(root_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output / "images").mkdir(exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    
    # Load file list
    with open(split_file, 'r') as f:
        paths = [line.strip() for line in f]
    
    crop_info = []
    crop_id = 0
    
    print(f"Processing {len(paths)} images...")
    
    for stem in tqdm(paths, desc="Extracting crops"):
        # Load full image and mask
        img_path = root / f"Images/{stem}{img_ext}"
        mask_path = root / f"Masks/{stem}{mask_ext}"
        
        if not img_path.exists() or not mask_path.exists():
            print(f"Warning: Missing files for {stem}")
            continue
            
        img = Image.open(img_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask = rgb_to_index(mask_rgb)
        
        img_array = np.array(img)
        h_img, w_img = img_array.shape[:2]
        
        # Extract grid crops
        n_rows = h_img // crop_size
        n_cols = w_img // crop_size
        
        for row in range(n_rows):
            for col in range(n_cols):
                y = row * crop_size
                x = col * crop_size
                
                img_crop = img_array[y:y+crop_size, x:x+crop_size, :]
                mask_crop_rgb = mask_rgb[y:y+crop_size, x:x+crop_size, :]
                mask_crop = mask[y:y+crop_size, x:x+crop_size]
                
                # Check background ratio using RGB color (0, 0, 0) = Background
                bg_ratio = np.all(mask_crop_rgb == (0, 0, 0), axis=-1).mean()
                
                if bg_ratio <= max_bg_ratio:
                    # Save crop
                    crop_name = f"crop_{crop_id:06d}"
                    
                    img_crop_pil = Image.fromarray(img_crop)
                    mask_crop_pil = Image.fromarray(mask_crop)
                    
                    img_crop_pil.save(output / "images" / f"{crop_name}.png")
                    mask_crop_pil.save(output / "masks" / f"{crop_name}.png")
                    
                    # Store metadata
                    crop_info.append({
                        'crop_id': crop_id,
                        'stem': stem,
                        'row': row,
                        'col': col,
                        'bg_ratio': bg_ratio
                    })
                    
                    crop_id += 1
    
    # Save crop metadata
    metadata_file = output / "crop_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("crop_id,stem,row,col,bg_ratio\n")
        for info in crop_info:
            f.write(f"{info['crop_id']},{info['stem']},{info['row']},{info['col']},{info['bg_ratio']:.4f}\n")
    
    print(f"✅ Extracted {len(crop_info)} valid crops")
    print(f"📁 Saved to: {output}")
    print(f"📊 Metadata: {metadata_file}")
    
    return len(crop_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute dataset crops")
    parser.add_argument("--root", required=True, help="Path to dataset root")
    parser.add_argument("--split", required=True, help="Path to split file (train_files.txt/validation_files.txt)")
    parser.add_argument("--output", required=True, help="Output directory for crops")
    parser.add_argument("--img_ext", default=".tif", help="Image extension")
    parser.add_argument("--mask_ext", default=".png", help="Mask extension")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size")
    parser.add_argument("--max_bg_ratio", type=float, default=0.9, help="Maximum background ratio (exclude crops with >90% background)")
    
    args = parser.parse_args()
    
    extract_and_save_crops(
        args.root, args.split, args.output, 
        args.img_ext, args.mask_ext, args.crop_size, args.max_bg_ratio
    )