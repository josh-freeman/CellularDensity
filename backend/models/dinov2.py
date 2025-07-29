"""
Shared DINOv2 segmentation model architecture following Facebook's official implementation.
Used by both training and inference to ensure exact consistency.
"""

import math
import itertools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPadding(nn.Module):
    """Center padding to ensure input dimensions are multiples of patch size."""
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class BNHead(nn.Module):
    """Official DINOv2 segmentation head: BatchNorm + 1x1 Conv (like mmcv BNHead)."""
    def __init__(self, in_channels, num_classes, ignore_index=255):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Simple head: BatchNorm + 1x1 Conv
        self.bn = nn.SyncBatchNorm(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Initialize weights following official implementation
        self._init_weights()
    
    def _init_weights(self):
        # Initialize conv_seg with normal distribution (std=0.01)
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        if self.conv_seg.bias is not None:
            nn.init.constant_(self.conv_seg.bias, 0)
    
    def forward(self, inputs):
        """
        Args:
            inputs: List of feature maps from multiple scales
        """
        # Concatenate features from different scales
        if isinstance(inputs, (list, tuple)):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs
        
        # Apply BatchNorm + Conv
        x = self.bn(x)
        output = self.conv_seg(x)
        
        return output


class DINOv2Seg(nn.Module):
    """Official DINOv2 segmentation model following Facebook's implementation."""
    
    def __init__(self, model_name="vit_base_patch14_dinov2", n_classes=12, pretrained=True):
        super().__init__()
        
        # Load DINOv2 backbone using torch.hub (official method)
        backbone_size = self._get_backbone_size(model_name)
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14", 
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[backbone_size]
        backbone_hub_name = f"dinov2_{backbone_arch}"
        
        # Load backbone from torch.hub
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", 
            model=backbone_hub_name,
            pretrained=pretrained
        )
        self.backbone.eval()
        
        # Get patch size and determine layer indices
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.out_indices = self._get_out_indices(backbone_size)
        
        # Setup intermediate layer extraction (official method)
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=self.out_indices,
            reshape=True,
        )
        
        # Add center padding for proper patch alignment
        self.center_padding = CenterPadding(self.patch_size)
        
        # Calculate total feature dimension after concatenation
        embed_dim = self.backbone.embed_dim
        total_channels = embed_dim * len(self.out_indices)
        
        # Official segmentation head (BNHead)
        self.decode_head = BNHead(
            in_channels=total_channels,
            num_classes=n_classes
        )

    def _get_backbone_size(self, model_name):
        """Extract backbone size from model name."""
        if 'small' in model_name.lower():
            return "small"
        elif 'base' in model_name.lower():
            return "base"
        elif 'large' in model_name.lower():
            return "large"
        elif 'giant' in model_name.lower():
            return "giant"
        else:
            return "base"  # Default

    def _get_out_indices(self, backbone_size):
        """Get output indices for intermediate layer extraction."""
        # Based on official implementation
        if backbone_size == "small":
            return [2, 5, 8, 11]  # 12 layers total
        elif backbone_size == "base":
            return [2, 5, 8, 11]  # 12 layers total  
        elif backbone_size == "large":
            return [5, 11, 17, 23]  # 24 layers total
        elif backbone_size == "giant":
            return [9, 19, 29, 39]  # 40 layers total
        else:
            return [2, 5, 8, 11]  # Default

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Apply center padding for proper patch alignment
        x_padded = self.center_padding(x)
        
        # Extract intermediate features using official method
        # backbone.forward now calls get_intermediate_layers with reshape=True
        features = self.backbone(x_padded)  # List of [B, C, H', W'] features
        
        # Resize all features to same scale (largest feature map)
        target_size = features[0].shape[2:]  # Use first feature map size as target
        aligned_features = []
        
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            aligned_features.append(feat)
        
        # Apply segmentation head (concatenate + BN + Conv)
        output = self.decode_head(aligned_features)  # [B, num_classes, H', W']
        
        # Upsample to original resolution
        output = F.interpolate(
            output, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return output