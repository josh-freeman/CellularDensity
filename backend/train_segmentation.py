"""
train_segmentation.py
Fine-tunes various backbone models for pixel-wise segmentation of skin histopathology.
Supports multiple architectures including GigaPath, DINOv2, EfficientNet, and ResNet.
"""

import argparse, os, json, random, glob
from pathlib import Path
from typing import Union

import torch, timm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import wandb

import timm
from segmentation_models_pytorch.encoders import encoders
from segmentation_models_pytorch.encoders.timm_universal import TimmUniversalEncoder

import numpy as np
from PIL import Image

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
        # Create DINOv2 model with flexible input size
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=224  # Force 224x224 input size
        )

        # Get patch size and embedding dimension
        self.patch_h = self.patch_w = self.backbone.patch_embed.patch_size[0]
        C = self.backbone.embed_dim

        # UNet-style decoder
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


# --------------------- Dataset ----------------------------
class PatchDataset(Dataset):
    """
    Dataset that yields 224×224 crops with ≥10% foreground pixels.
    
    Each entry in train_files.txt must point to the tile stem,
    e.g., 10x/Images/case123_tile004 (without extension).
    The Dataset will:
      1. Load the full image + mask
      2. Sample a random 224×224 crop where >= min_fg_ratio pixels
         in the mask are non-zero
      3. Apply (optionally) Albumentations augmentation + normalization
    """
    def __init__(
        self,
        root: Union[str, Path],
        split_txt: str,
        img_ext: str = ".png",
        mask_ext: str = ".png",
        augment: bool = False,
        crop_size: int = 224,
        min_fg_ratio: float = 0.10,
        max_tries: int = 10,
    ):
        self.root = Path(root)
        self.paths = [p.strip() for p in open(split_txt)]
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.crop_size = crop_size
        self.min_fg_ratio = min_fg_ratio
        self.max_tries = max_tries

        # Albumentations pipeline
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.t_aug = (A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean, std),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ]))

    def _load_pair(self, stem):
        img_path = self.root / f"Images/{stem}{self.img_ext}"
        mask_path = self.root / f"Masks/{stem}{self.mask_ext}"
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("RGB"))  # keep color
        mask = rgb_to_index(mask)  # convert to 0-11 / 255
        return np.array(img), mask

    def _random_crop(self, img: np.ndarray, mask: np.ndarray):
        """
        Random 224×224 crop with ≥ min_fg_ratio foreground pixels.
        img  : H×W×3  uint8
        mask : H×W    uint8
        """
        h_img, w_img = img.shape[:2]
        cs = self.crop_size
        best_crop = None
        best_fg = -1.0

        for _ in range(self.max_tries):
            x = random.randint(0, w_img - cs)
            y = random.randint(0, h_img - cs)
            img_crop = img[y:y+cs, x:x+cs, :]
            mask_crop = mask[y:y+cs, x:x+cs]

            fg_ratio = (mask_crop != IGNORE_LABEL).mean()
            if fg_ratio > best_fg:
                best_fg, best_crop = fg_ratio, (img_crop, mask_crop)
            if fg_ratio >= self.min_fg_ratio:
                break

        return best_crop  # (img_crop, mask_crop)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        stem = self.paths[idx]
        img, mask = self._load_pair(stem)
        img, mask = self._random_crop(img, mask)  # enforce ≥10% FG

        # Albumentations expects numpy arrays
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
    wandb_name = f"{backbone_short}-histoseg-{args.epochs}ep-bs{args.bs}-lr{args.lr}"
    if resume_from_checkpoint:
        wandb_name += "-resumed"
        
    wandb.init(
        project="skin-histopathology-segmentation",
        name=wandb_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "freeze_encoder_epochs": args.freeze_encoder_epochs,
            "n_classes": args.n_classes,
            "backbone": args.backbone,
            "backbone_display": backbone_short,
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
    
    # Datasets
    train_ds = PatchDataset(args.root, f"{args.root}/../train_files.txt", augment=True, img_ext=args.img_ext, mask_ext=args.mask_ext)
    val_ds = PatchDataset(args.root, f"{args.root}/../validation_files.txt", img_ext=args.img_ext, mask_ext=args.mask_ext)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=16, pin_memory=True, collate_fn=simple_collate, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=8, collate_fn=simple_collate, persistent_workers=True)

    # Model
    model = build_model(backbone=args.backbone, out_classes=args.n_classes).to(args.device)
    enc = model.encoder if hasattr(model, "encoder") else getattr(model, "backbone", None)
    
    # Freeze encoder initially
    freeze_encoder = args.freeze_encoder_epochs > 0  
    for p in enc.parameters():
        p.requires_grad = not freeze_encoder

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
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
        
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(args.device), masks.to(args.device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / max(num_batches, 1)  # Avoid division by zero

        # Unfreeze after warm-up
        if epoch == args.freeze_encoder_epochs + 1:
            for p in enc.parameters():
                p.requires_grad = True
            already_in_opt = set(id(p) for g in opt.param_groups for p in g["params"])
            opt.add_param_group(
                {"params": [p for p in enc.parameters() if id(p) not in already_in_opt]}
            )
            print(f"🔓 Encoder unfrozen at epoch {epoch}")

        # Validation
        model.eval()
        vloss, n = 0, 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(args.device), masks.to(args.device)
                vloss += criterion(model(imgs), masks).item() * imgs.size(0)
                n += imgs.size(0)
        vloss /= n
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": vloss,
            "learning_rate": opt.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch:02d}: train_loss {avg_train_loss:.4f}, val_loss {vloss:.4f}")
        
        # Save best model with backbone-specific naming
        if vloss < best_val:
            best_val = vloss
            model_filename = f"{args.backbone}_unet_best.pt"
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
  # Train with GigaPath backbone (default):
  python train_segmentation.py --root /path/to/dataset/data
  
  # Train with specific backbone:
  python train_segmentation.py --root /path/to/dataset/data --backbone efficientnet-b5
  
  # Resume training from checkpoint:
  python train_segmentation.py --root /path/to/dataset/data --resume_from_checkpoint auto
  
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