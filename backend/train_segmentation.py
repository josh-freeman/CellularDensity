"""
train_segmentation.py
Fine-tunes various backbone models for pixel-wise segmentation of skin histopathology.
Supports multiple architectures including GigaPath, DINOv2, EfficientNet, and ResNet.

Key features:
- Grid-based non-overlapping crop extraction (no random cropping)
- Systematic coverage of entire training images
- Multiple backbone architecture support with optimized defaults
"""

import argparse, os, json, glob, shutil
from pathlib import Path
from typing import Union

import torch, timm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import wandb

import numpy as np
from PIL import Image

# Import precompute function
from precompute_crops import extract_and_save_crops

# ---------------------------------- Color map ---------------------------------
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
    """
    Convert RGB mask to index mask.
    mask_rgb : H×W×3 uint8  →  H×W uint8 with values 0-11 (or 255 if unknown)
    """
    idx_mask = np.full(mask_rgb.shape[:2], IGNORE_LABEL, dtype=np.uint8)
    for rgb, idx in COLOR2IDX.items():
        matches = np.all(mask_rgb == rgb, axis=-1)
        idx_mask[matches] = idx
    return idx_mask


# Custom model implementations for GigaPath and DINOv2
class GigaPathSeg(nn.Module):
    def __init__(self, n_classes=12, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=pretrained,
            num_classes=0
        )

        # Handle both tuple and int patch sizes
        _p = self.backbone.patch_embed.patch_size
        self.patch_h, self.patch_w = (_p, _p) if isinstance(_p, int) else _p
        C = self.backbone.embed_dim  # 1024

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(C, 512, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.GELU(),
            nn.Conv2d(128, n_classes, 1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        tokens = self.backbone.forward_features(x)  # B, N+1, C
        tokens = tokens[:, 1:, :]  # drop CLS token

        # Reshape sequence back to 2-D grid
        h = H // self.patch_h
        w = W // self.patch_w
        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)  # B, C, h, w

        mask = self.decoder(feat)
        return nn.functional.interpolate(
            mask, size=(H, W), mode="bilinear", align_corners=False
        )


class DINOv2Seg(nn.Module):
    def __init__(self, model_name="vit_base_patch14_dinov2", n_classes=12, pretrained=True):
        super().__init__()
        # Create DINOv2 model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=224  # Force 224x224 input size
        )

        # Get patch size and embedding dimension
        self.patch_h = self.patch_w = self.backbone.patch_embed.patch_size[0]
        C = self.backbone.embed_dim
        
        # Determine which layers to extract features from
        if 'small' in model_name:
            self.layer_indices = [2, 5, 8, 11]  # 12 layers total
        elif 'base' in model_name:
            self.layer_indices = [2, 5, 8, 11]  # 12 layers total
        elif 'large' in model_name:
            self.layer_indices = [5, 11, 17, 23]  # 24 layers total
        elif 'giant' in model_name:
            self.layer_indices = [9, 19, 29, 39]  # 40 layers total
        else:
            self.layer_indices = [2, 5, 8, 11]  # Default
        
        # Multi-scale feature fusion with FPN-like architecture
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        fusion_dim = 256
        
        for _ in self.layer_indices:
            # Lateral connections to reduce channel dimensions
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(C, fusion_dim, kernel_size=1),
                    nn.BatchNorm2d(fusion_dim),
                    nn.ReLU(inplace=True)
                )
            )
            # FPN convolutions for feature refinement
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(fusion_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Sophisticated decoder head with ASPP-like module
        self.decoder = nn.Sequential(
            # ASPP-style multi-scale processing
            ASPPModule(fusion_dim * len(self.layer_indices), 512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        
        # Initialize decoder weights
        self._init_weights()
        
        # Register hooks to extract intermediate features
        self.intermediate_features = []
        self._register_hooks()

    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.intermediate_features.append(output)
        
        # Register hooks on specified transformer blocks
        for idx in self.layer_indices:
            self.backbone.blocks[idx].register_forward_hook(hook_fn)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Clear previous features
        self.intermediate_features = []
        
        # Forward pass through backbone (this will trigger hooks)
        _ = self.backbone.forward_features(x)
        
        # Process intermediate features with FPN
        laterals = []
        for i, (feat, lateral_conv, fpn_conv) in enumerate(zip(
            self.intermediate_features, self.lateral_convs, self.fpn_convs
        )):
            # Each feature is still in sequence format (B, N+1, C)
            # Remove CLS token and reshape to 2D
            feat = feat[:, 1:, :]  # B, N, C
            
            # Calculate spatial dimensions
            h = H // self.patch_h
            w = W // self.patch_w
            
            # Reshape to spatial feature map
            feat = feat.transpose(1, 2).reshape(B, -1, h, w)  # B, C, h, w
            
            # Apply lateral connection
            lateral = lateral_conv(feat)
            
            # Add top-down connection for FPN (except for the first level)
            if i > 0:
                # Upsample previous level and add
                prev_shape = lateral.shape[2:]
                top_down = nn.functional.interpolate(
                    laterals[-1], size=prev_shape, mode='nearest'
                )
                lateral = lateral + top_down
            
            # Apply FPN convolution
            lateral = fpn_conv(lateral)
            laterals.append(lateral)
        
        # Resize all features to the same scale (largest feature map size)
        target_size = laterals[0].shape[2:]
        aligned_features = []
        for lateral in laterals:
            if lateral.shape[2:] != target_size:
                lateral = nn.functional.interpolate(
                    lateral, size=target_size, mode='bilinear', align_corners=False
                )
            aligned_features.append(lateral)
        
        # Concatenate multi-scale features
        fused_features = torch.cat(aligned_features, dim=1)
        
        # Apply decoder
        masks = self.decoder(fused_features)
        
        # Upsample to original resolution
        masks = nn.functional.interpolate(
            masks, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return masks


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module for multi-scale context aggregation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple parallel convolutions with different dilation rates
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + out_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Apply parallel convolutions
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        # Global context
        feat5 = self.gap(x)
        feat5 = nn.functional.interpolate(feat5, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        
        # Final fusion
        out = self.fusion(out)
        
        return out


# --------------------- Dataset ----------------------------
class PatchDataset(Dataset):
    """
    Fast dataset that loads precomputed 224×224 crops directly from disk.
    Expects crops to be precomputed using precompute_crops.py.
    """
    def __init__(
        self,
        crop_dir: Union[str, Path],
        augment: bool = False,
    ):
        self.crop_dir = Path(crop_dir)
        self.augment = augment
        
        # Load crop metadata
        metadata_file = self.crop_dir / "crop_metadata.txt"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Crop metadata not found: {metadata_file}")
        
        self.crop_list = []
        with open(metadata_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                crop_id = int(parts[0])
                self.crop_list.append(crop_id)
        
        # Albumentations pipeline - geometric + brightness/contrast (no stretching)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.t_aug = (A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),  # Increased probability for 90-degree rotations
            A.Transpose(p=0.5),  # Additional reflection (transpose = flip along diagonal)
            A.RandomBrightnessContrast(p=0.3),  # Keep brightness/contrast for staining variation
            A.Normalize(mean, std),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ]))
        
        print(f"✅ Found {len(self.crop_list)} precomputed crops in {crop_dir}")

    def __len__(self):
        return len(self.crop_list)

    def __getitem__(self, idx):
        crop_id = self.crop_list[idx]
        crop_name = f"crop_{crop_id:06d}"
        
        # Load precomputed crop files
        img_path = self.crop_dir / "images" / f"{crop_name}.png"
        mask_path = self.crop_dir / "masks" / f"{crop_name}.png"
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # Apply augmentations
        aug = self.t_aug(image=img, mask=mask)
        return aug["image"], aug["mask"].long()


def simple_collate(batch):
    """Simple collate function for DataLoader."""
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)


# --------------------- Model Building -------------------------------
def load_model_configs(config_file="model_configs.json"):
    """Load model configurations from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file {config_file} not found, using built-in defaults")
        # Fallback to built-in defaults if file doesn't exist
        return {
            "model_defaults": {},
            "default_fallback": {
                "lr": 1e-4,
                "freeze_encoder_epochs": 2,
                "batch_size": 16,
                "weight_decay": 1e-4,
                "description": "Generic defaults"
            },
            "model_groups": {}
        }
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing {config_file}: {e}")
        print("💡 Check the JSON syntax in your config file")
        raise


def get_model_defaults(backbone: str, config_file="model_configs.json"):
    """Get model-specific default hyperparameters from config file."""
    config = load_model_configs(config_file)
    model_defaults = config.get("model_defaults", {})
    default_fallback = config.get("default_fallback", {
        "lr": 1e-4,
        "freeze_encoder_epochs": 2,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "description": "Generic defaults"
    })
    
    return model_defaults.get(backbone, default_fallback)


def build_model(backbone: str = "gigapath_vitl", out_classes: int = 12):
    """
    Build segmentation model with specified backbone.
    Returns encoder + UNet decoder.
    """
    # Map of backbone display names for logging
    backbone_info = {
        "gigapath_vitl": "Prov-GigaPath ViT-Large (1.3B histopathology tiles)",
        "resnet50": "ResNet-50 (ImageNet pretrained)",
        "efficientnet-b3": "EfficientNet-B3 (ImageNet pretrained)",
        "efficientnet-b5": "EfficientNet-B5 (ImageNet pretrained)",
        "efficientnet-b7": "EfficientNet-B7 (ImageNet pretrained)",
        "resnext50_32x4d": "ResNeXt-50 (ImageNet pretrained)",
        "resnet34": "ResNet-34 (ImageNet pretrained)",
        "resnet101": "ResNet-101 (ImageNet pretrained)",
        "densenet121": "DenseNet-121 (ImageNet pretrained)",
        "mobilenet_v2": "MobileNet-V2 (ImageNet pretrained)",
        "vit_small_patch14_dinov2": "DINOv2 ViT-Small/14 (Self-supervised)",
        "vit_base_patch14_dinov2": "DINOv2 ViT-Base/14 (Self-supervised)", 
        "vit_large_patch14_dinov2": "DINOv2 ViT-Large/14 (Self-supervised)",
        "vit_giant_patch14_dinov2": "DINOv2 ViT-Giant/14 (Self-supervised)",
    }
    
    print(f"🏗️  Building model with backbone: {backbone}")
    if backbone in backbone_info:
        print(f"   📝 {backbone_info[backbone]}")
    
    # Special handling for GigaPath - use original custom implementation
    if backbone == "gigapath_vitl":
        try:
            model = GigaPathSeg(n_classes=out_classes, pretrained=True)
            print(f"✅ Successfully created GigaPath custom model")
            return model
        except Exception as e:
            print(f"❌ Error creating GigaPath model: {e}")
            raise
    
    # Special handling for DINOv2 models - use custom implementation
    if 'dinov2' in backbone:
        try:
            model = DINOv2Seg(model_name=backbone, n_classes=out_classes, pretrained=True)
            print(f"✅ Successfully created DINOv2 custom model")
            return model
        except Exception as e:
            print(f"❌ Error creating DINOv2 model: {e}")
            raise
    
    # For all other backbones, use segmentation-models-pytorch
    try:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",  # Standard ImageNet pretraining
            classes=out_classes,
            activation=None
        )
        print(f"✅ Successfully created SMP model with {backbone} backbone")
        return model
        
    except Exception as e:
        print(f"❌ Error creating model with backbone '{backbone}': {e}")
        print(f"💡 Available backbones include: gigapath_vitl, resnet50, efficientnet-b3, resnext50_32x4d")
        print(f"💡 Try: resnet34, resnet101, densenet121, mobilenet_v2, efficientnet-b5, efficientnet-b7, vit_base_patch14_dinov2")
        raise


# --------------------- Training Functions --------------------------
def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """Find the most recent checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    def get_epoch_num(filename):
        return int(filename.split('_epoch_')[1].split('.pt')[0])
    
    latest_checkpoint = max(checkpoint_files, key=get_epoch_num)
    return latest_checkpoint


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="./checkpoints"):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Keep only the last 5 checkpoints to save space
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if len(checkpoint_files) > 5:
        # Sort by modification time to keep the most recent
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
        for old_checkpoint in checkpoint_files[:-5]:
            print(f"🗑️  Removing old checkpoint: {os.path.basename(old_checkpoint)}")
            os.remove(old_checkpoint)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if this is a full checkpoint or just model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✅ Resumed from full checkpoint: epoch {epoch}, best val loss: {val_loss:.4f}")
        return epoch, val_loss
        
    else:
        # Just model weights
        model.load_state_dict(checkpoint)
        
        print("⚠️  Loaded model weights only (no training metadata)")
        print("   Starting from epoch 1 with fresh optimizer state")
        print("   Note: This is like transfer learning from the best model")
        
        return 0, float('inf')  # Start fresh with epoch 0, unknown best loss


def dice_loss(pred, target, eps=1e-6, ignore_index=255):
    """
    Dice loss for segmentation.
    pred : B×C×H×W  (logits)
    target : B×H×W  (uint8 indices 0-11 or 255)
    """
    # Mask out ignored pixels
    valid = (target != ignore_index)
    if valid.sum() == 0:  # all ignore? avoid NaN
        return pred.sum() * 0.0

    pred = pred.permute(0, 2, 3, 1)[valid]  # → N_valid × C
    target = target[valid]  # → N_valid
    target_onehot = torch.nn.functional.one_hot(target, pred.size(-1)).float()

    pred = torch.softmax(pred, -1)

    intersection = (pred * target_onehot).sum(0)
    union = pred.sum(0) + target_onehot.sum(0)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train(args):
    """Main training function."""
    # Apply model-specific defaults if not explicitly set by user
    model_defaults = get_model_defaults(args.backbone)
    
    # Get gradient accumulation steps from model defaults
    gradient_accumulation_steps = model_defaults.get("gradient_accumulation_steps", 1)
    
    # Override defaults only if user didn't specify
    if not hasattr(args, '_lr_set'):
        args.lr = model_defaults["lr"]
        print(f"🔧 Using model-specific learning rate: {args.lr}")
    
    if not hasattr(args, '_bs_set'):
        args.bs = model_defaults["batch_size"] 
        print(f"🔧 Using model-specific batch size: {args.bs}")
    
    if not hasattr(args, '_freeze_set'):
        args.freeze_encoder_epochs = model_defaults["freeze_encoder_epochs"]
        print(f"🔧 Using model-specific freeze epochs: {args.freeze_encoder_epochs}")
    
    if not hasattr(args, '_epochs_set') and 'default_epochs' in model_defaults:
        args.epochs = model_defaults["default_epochs"]
        print(f"🔧 Using model-specific epochs: {args.epochs}")
    
    print(f"💡 Model defaults: {model_defaults['description']}")
    
    # Handle checkpoint resumption
    start_epoch = 1
    best_val = 1e9
    resume_from_checkpoint = False
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            # Find the latest checkpoint automatically
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
            if checkpoint_path:
                print(f"🔄 Auto-resuming from latest checkpoint: {checkpoint_path}")
                resume_from_checkpoint = True
            else:
                print("⚠️  No checkpoints found for auto-resume, starting fresh")
        else:
            # Use specific checkpoint path
            checkpoint_path = args.resume_from_checkpoint
            if os.path.exists(checkpoint_path):
                print(f"🔄 Resuming from specified checkpoint: {checkpoint_path}")
                resume_from_checkpoint = True
            else:
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Initialize wandb with descriptive run name
    backbone_display = {
        "gigapath_vitl": "GigaPath-ViTL",
        "resnet50": "ResNet50",
        "resnet34": "ResNet34", 
        "resnet101": "ResNet101",
        "efficientnet-b3": "EfficientNet-B3",
        "efficientnet-b5": "EfficientNet-B5",
        "efficientnet-b7": "EfficientNet-B7",
        "resnext50_32x4d": "ResNeXt50",
        "densenet121": "DenseNet121",
        "mobilenet_v2": "MobileNet-V2",
        "vit_small_patch14_dinov2": "DINOv2-ViTS14",
        "vit_base_patch14_dinov2": "DINOv2-ViTB14", 
        "vit_large_patch14_dinov2": "DINOv2-ViTL14",
        "vit_giant_patch14_dinov2": "DINOv2-ViTG14"
    }
    
    backbone_short = backbone_display.get(args.backbone, args.backbone)
    wandb_name = f"{backbone_short}-histoseg-{args.magnification}-{args.epochs}ep-bs{args.bs}-lr{args.lr}"
    if resume_from_checkpoint:
        wandb_name += "-resumed"
        
    wandb.init(
        project="skin-histopathology-segmentation",
        name=wandb_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.bs,
            "effective_batch_size": args.bs * gradient_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": args.lr,
            "freeze_encoder_epochs": args.freeze_encoder_epochs,
            "n_classes": args.n_classes,
            "backbone": args.backbone,
            "backbone_display": backbone_short,
            "magnification": args.magnification,
            "img_ext": args.img_ext,
            "mask_ext": args.mask_ext,
            "architecture": f"{backbone_short}-UNet",
            "dataset": "skin-histopathology-segmentation",
            "domain": "histopathology" if args.backbone == "gigapath_vitl" else "natural-images",
            "model_params": "1.1B" if args.backbone == "gigapath_vitl" else "varies",
            "resumed": resume_from_checkpoint
        }
    )
    
    print(f"Started wandb run: {wandb.run.name}")
    print(f"Wandb run URL: {wandb.run.url}")
    
    # Datasets - use precomputed crops
    train_crop_dir = f"{args.root}_crops_train"
    val_crop_dir = f"{args.root}_crops_val"
    
    # Handle crop precomputation
    def run_precompute(root, split_file, output_dir, force=False):
        """Run precompute_crops directly if needed."""
        if force and Path(output_dir).exists():
            print(f"🗑️  Removing existing crops: {output_dir}")
            shutil.rmtree(output_dir)
        
        if not Path(output_dir).exists() or not (Path(output_dir) / "crop_metadata.txt").exists():
            print(f"🔄 Precomputing crops: {output_dir}")
            try:
                extract_and_save_crops(
                    root_dir=root,
                    split_file=split_file,
                    output_dir=output_dir,
                    img_ext=args.img_ext,
                    mask_ext=args.mask_ext,
                    crop_size=224,
                    max_bg_ratio=args.max_bg_ratio
                )
                print(f"✅ Precomputation completed: {output_dir}")
            except Exception as e:
                print(f"❌ Precomputation failed: {e}")
                import sys
                sys.exit(1)
        else:
            print(f"✅ Using existing crops: {output_dir}")
    
    # Run precomputation for train and validation sets
    run_precompute(args.root, f"{args.root}/../train_files.txt", train_crop_dir, args.force_precompute)
    run_precompute(args.root, f"{args.root}/../validation_files.txt", val_crop_dir, args.force_precompute)
    
    train_ds = PatchDataset(train_crop_dir, augment=True)
    val_ds = PatchDataset(val_crop_dir, augment=False)

    # Optimize data loading for better GPU utilization
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=simple_collate,
                          persistent_workers=True, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=1, pin_memory=True, collate_fn=simple_collate,
                        persistent_workers=True, prefetch_factor=2)

    # Model
    print(f"🎯 Using device: {args.device}")
    print(f"🎯 CUDA available: {torch.cuda.is_available()}")
    print(f"🎯 CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU name: {torch.cuda.get_device_name()}")
        print(f"🎯 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    model = build_model(backbone=args.backbone, out_classes=args.n_classes).to(args.device)
    enc = model.encoder if hasattr(model, "encoder") else getattr(model, "backbone", None)
    
    # Verify model is on GPU
    print(f"🎯 Model device: {next(model.parameters()).device}")
    
    # Freeze encoder initially
    freeze_encoder = args.freeze_encoder_epochs > 0  
    for p in enc.parameters():
        p.requires_grad = not freeze_encoder

    # Optimizer configuration
    if 'dinov2' in args.backbone:
        # For DINOv2, use different learning rates for backbone and head
        param_groups = []
        
        # Head parameters with full learning rate
        if hasattr(model, 'head'):
            param_groups.append({'params': model.head.parameters(), 'lr': args.lr})
        else:
            # Fallback for other decoder architectures
            decoder_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
            param_groups.append({'params': decoder_params, 'lr': args.lr})
        
        # Backbone parameters with reduced learning rate (when unfrozen)
        if not freeze_encoder:
            param_groups.append({'params': enc.parameters(), 'lr': args.lr * 0.1})
        
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=1e-4)
    
    # Loss function - use CrossEntropy as primary loss for DINOv2
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    criterion = dice_loss

    # Load checkpoint if resuming
    if resume_from_checkpoint:
        start_epoch, best_val = load_checkpoint(checkpoint_path, model, opt)
        start_epoch += 1  # Start from next epoch
        
        # Re-evaluate encoder freezing based on resumed epoch
        current_epoch = start_epoch - 1
        if current_epoch > args.freeze_encoder_epochs:
            # Unfreeze encoder if we're past the freeze period
            for p in enc.parameters():
                p.requires_grad = True
            # Re-create optimizer to include all parameters
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            print(f"🔓 Encoder unfrozen (resumed past freeze period)")
        else:
            print(f"🔒 Encoder still frozen (resumed epoch {current_epoch} <= freeze epochs {args.freeze_encoder_epochs})")
            
        print(f"📊 Resuming training from epoch {start_epoch} with best val loss: {best_val:.4f}")
    else:
        print(f"🆕 Starting fresh training")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Zero gradients at the start of epoch
        opt.zero_grad()
        
        for batch_idx, (imgs, masks) in enumerate(train_dl):
            imgs, masks = imgs.to(args.device), masks.to(args.device)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"🎯 Batch {batch_idx}: imgs.device = {imgs.device}, masks.device = {masks.device}")
                print(f"🎯 Batch shape: imgs={imgs.shape}, masks={masks.shape}")
            
            out = model(imgs)
            
            # For DINOv2, use pure CrossEntropy loss like Facebook's implementation
            if 'dinov2' in args.backbone:
                loss = ce_loss(out, masks)
            else:
                loss = criterion(out, masks)
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Step optimizer every gradient_accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dl):
                # Gradient clipping for stability with ViT models
                if 'dinov2' in args.backbone or 'gigapath' in args.backbone:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                opt.step()
                opt.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
            num_batches += 1
        
        avg_train_loss = epoch_loss / max(num_batches, 1)  # Avoid division by zero

        # Unfreeze after warm-up
        if epoch == args.freeze_encoder_epochs + 1 and args.freeze_encoder_epochs > 0:
            for p in enc.parameters():
                p.requires_grad = True
            
            if 'dinov2' in args.backbone:
                # Add encoder parameters with much lower learning rate
                opt.add_param_group({
                    'params': enc.parameters(), 
                    'lr': args.lr * 0.01  # 1% of head learning rate
                })
                print(f"🔓 DINOv2 encoder unfrozen at epoch {epoch} with LR={args.lr * 0.01}")
            else:
                already_in_opt = set(id(p) for g in opt.param_groups for p in g["params"])
                opt.add_param_group(
                    {"params": [p for p in enc.parameters() if id(p) not in already_in_opt]}
                )
                print(f"🔓 Encoder unfrozen at epoch {epoch}")

        # Validation
        model.eval()
        vloss, vce, n = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(args.device), masks.to(args.device)
                out = model(imgs)
                
                # Calculate losses
                if 'dinov2' in args.backbone:
                    # For DINOv2, primary metric is CE loss
                    ce_loss_val = ce_loss(out, masks)
                    dice_loss_val = criterion(out, masks)
                    vloss += ce_loss_val.item() * imgs.size(0)  # Track CE as primary
                    vce += ce_loss_val.item() * imgs.size(0)
                else:
                    # For other models, primary metric is Dice loss
                    dice_loss_val = criterion(out, masks)
                    ce_loss_val = ce_loss(out, masks)
                    vloss += dice_loss_val.item() * imgs.size(0)  # Track Dice as primary
                    vce += ce_loss_val.item() * imgs.size(0)
                
                n += imgs.size(0)
        
        vloss /= n
        vce /= n
        
        # Log metrics to wandb
        log_dict = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": vloss,  # Primary metric (CE for DINOv2, Dice for others)
            "val_ce_loss": vce,
            "val_dice_loss": vloss if not 'dinov2' in args.backbone else None,
            "learning_rate": opt.param_groups[0]['lr']
        }
        
        wandb.log(log_dict)
        
        # Print appropriate metrics based on model type
        if 'dinov2' in args.backbone:
            print(f"Epoch {epoch:02d}: train_loss {avg_train_loss:.4f}, val_ce_loss {vloss:.4f}")
        else:
            print(f"Epoch {epoch:02d}: train_loss {avg_train_loss:.4f}, val_dice_loss {vloss:.4f}")
        
        # Save best model with backbone-specific naming
        if vloss < best_val:
            best_val = vloss
            model_filename = f"{args.backbone}_{args.magnification}_unet_best.pt"
            torch.save(model.state_dict(), model_filename)
            wandb.save(model_filename)
            wandb.log({"best_val_loss": best_val})
            print(f"💾 Best model saved as: {model_filename}")
        
        # Save checkpoint if enabled
        if args.save_checkpoints:
            should_save_checkpoint = False
            
            # Save checkpoint if this is the best model
            if vloss < best_val:
                should_save_checkpoint = True
                checkpoint_type = "best"
            # Or save every N epochs
            elif epoch % args.checkpoint_interval == 0:
                should_save_checkpoint = True
                checkpoint_type = "interval"
            
            if should_save_checkpoint:
                checkpoint_path = save_checkpoint(model, opt, epoch, vloss, args.checkpoint_dir)
                print(f"💾 Checkpoint saved ({checkpoint_type}): {checkpoint_path}")
                
                # Save to wandb only for best models or every 10 epochs
                if checkpoint_type == "best" or epoch % 10 == 0:
                    wandb.save(checkpoint_path)


# --------------------- CLI ------------------------------
class StoreAndMarkAction(argparse.Action):
    """Custom action to track when user explicitly sets a value."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f'_{self.dest}_set', True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train segmentation models for skin histopathology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with GigaPath backbone on 10x data:
  python train_segmentation.py --root /path/to/dataset/data/10x --magnification 10x
  
  # Train with specific backbone on 1x data:
  python train_segmentation.py --root /path/to/dataset/data/1x --backbone efficientnet-b5 --magnification 1x
  
  # Resume training from checkpoint:
  python train_segmentation.py --root /path/to/dataset/data/10x --magnification 10x --resume_from_checkpoint auto
  
  # Show available model defaults:
  python train_segmentation.py --show_defaults
        """
    )
    
    p.add_argument("--root", help="Path to dataset root directory (containing Images/ and Masks/ folders)")
    p.add_argument("--device", default="cuda", help="Device to use for training (cuda/cpu)")
    p.add_argument("--epochs", type=int, default=20, action=StoreAndMarkAction,
                   help="Number of training epochs (model-specific defaults will be used if not specified)")
    p.add_argument("--freeze_encoder_epochs", type=int, default=3, action=StoreAndMarkAction,
                   help="Epochs to freeze encoder (model-specific defaults will be used if not specified)")
    p.add_argument("--bs", type=int, default=16, action=StoreAndMarkAction,
                   help="Batch size (model-specific defaults will be used if not specified)")
    p.add_argument("--lr", type=float, default=3e-5, action=StoreAndMarkAction,
                   help="Learning rate (model-specific defaults will be used if not specified)")
    p.add_argument("--n_classes", type=int, default=12, help="Number of segmentation classes")
    p.add_argument("--backbone", default="gigapath_vitl", 
                   help='Backbone model to use. Options: gigapath_vitl (default), resnet50, resnet34, resnet101, '
                        'efficientnet-b3, efficientnet-b5, efficientnet-b7, resnext50_32x4d, densenet121, '
                        'mobilenet_v2, vit_small_patch14_dinov2, vit_base_patch14_dinov2, '
                        'vit_large_patch14_dinov2, vit_giant_patch14_dinov2')
    p.add_argument("--img_ext", default=".tif", help="Image file extension")
    p.add_argument("--mask_ext", default=".png", help="Mask file extension")
    p.add_argument("--magnification", type=str, required=True,
                   help='Magnification level of training data (e.g., "1x", "2x", "5x", "10x")')
    p.add_argument("--resume_from_checkpoint", default=None, 
                   help='Resume from checkpoint. Use "auto" for latest checkpoint, or provide specific path')
    p.add_argument("--checkpoint_dir", default="./checkpoints",
                   help="Directory to save/load checkpoints")
    p.add_argument("--save_checkpoints", action="store_true",
                   help="Save training checkpoints (best model + periodic saves)")
    p.add_argument("--checkpoint_interval", type=int, default=10,
                   help="Save checkpoint every N epochs (default: 10)")
    p.add_argument("--show_defaults", action="store_true",
                   help="Show model-specific default hyperparameters for all backbones and exit")
    p.add_argument("--force_precompute", action="store_true",
                   help="Force re-precomputation of crops (deletes existing crop directories)")
    p.add_argument("--max_bg_ratio", type=float, default=0.9,
                   help="Maximum background ratio for crop filtering (default: 0.9)")
    
    args = p.parse_args()
    
    # Show defaults and exit if requested
    if args.show_defaults:
        print("📋 Model-specific default hyperparameters:")
        print("=" * 80)
        print(f"📄 Config file: model_configs.json")
        
        # Load config and get groups
        config = load_model_configs()
        backbone_groups = config.get("model_groups", {
            "Available Models": list(config.get("model_defaults", {}).keys())
        })
        
        for group_name, backbones in backbone_groups.items():
            print(f"\n🔸 {group_name}:")
            for backbone in backbones:
                defaults = get_model_defaults(backbone)
                print(f"  {backbone}:")
                print(f"    Learning Rate: {defaults['lr']}")
                print(f"    Batch Size: {defaults['batch_size']}")
                print(f"    Freeze Epochs: {defaults['freeze_encoder_epochs']}")
                if 'default_epochs' in defaults:
                    print(f"    Default Epochs: {defaults['default_epochs']}")
                print(f"    Description: {defaults['description']}")
        
        print(f"\n💡 Usage: python train_segmentation.py --root /path/to/data --backbone vit_base_patch14_dinov2")
        print(f"💡 Override defaults: --lr 1e-4 --bs 32 --freeze_encoder_epochs 2")
        print(f"🔧 Edit model_configs.json to customize defaults")
        import sys
        sys.exit(0)
    
    # Check that root is provided for actual training
    if not args.root:
        print("❌ Error: --root is required for training")
        print("💡 Use --show_defaults to see model-specific hyperparameters")
        import sys
        sys.exit(1)
    
    try:
        train(args)
    finally:
        wandb.finish()