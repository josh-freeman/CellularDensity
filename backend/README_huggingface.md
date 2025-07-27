# 🔬 Skin Histopathology Segmentation Models

State-of-the-art deep learning models for 12-class skin histopathology segmentation, trained on high-resolution skin tissue images.

## 📋 Available Models

### EfficientNet Models
- **`efficientnet-b3_10x_unet_best.pt`** - EfficientNet-B3 backbone trained on 10x magnification data
- **`efficientnet-b7_10x_unet_best.pt`** - EfficientNet-B7 backbone trained on 10x magnification data

### Other Architectures
- **`efficientnet-b5_unet_best.pt`** - EfficientNet-B5 backbone (general purpose)
- **`gigapath_unet_best.pt`** - GigaPath vision transformer backbone

## 🏷️ Model Nomenclature

**Important**: All models now follow strict magnification-aware nomenclature:
- `{backbone}_{magnification}_unet_best.pt` for magnification-specific models
- Models without magnification suffix are general-purpose

This prevents silent failures where a model trained on one magnification is used with data from another magnification.

## 🔬 Tissue Classes

The models segment 12 distinct tissue classes:

| Class ID | Abbreviation | Full Name | Description |
|----------|--------------|-----------|-------------|
| 0 | GLD | Gland | Skin glandular structures |
| 1 | INF | Inflammation | Inflammatory tissue |
| 2 | FOL | Follicle | Hair follicles |
| 3 | HYP | Hypodermis | Subcutaneous tissue |
| 4 | RET | Reticular | Reticular dermis |
| 5 | PAP | Papillary | Papillary dermis |
| 6 | EPI | Epidermis | Outer skin layer |
| 7 | KER | Keratin | Keratinized tissue |
| 8 | BKG | Background | Background/non-tissue |
| 9 | BCC | Basal Cell Carcinoma | Cancer tissue |
| 10 | SCC | Squamous Cell Carcinoma | Cancer tissue |
| 11 | IEC | Inflammatory/Epithelial Cells | Mixed cell types |

## 🚀 Quick Start

### Using the Inference Script

```bash
# Download and use EfficientNet-B3 10x model
python skin_seg_inference.py image.jpg --model_name efficientnet-b3_10x

# Use with specific magnification parameter
python skin_seg_inference.py image.jpg --model_name efficientnet-b3 --magnification 10x

# Process whole slide images (NDPI, SVS, etc.)
python skin_seg_inference.py slide.ndpi --model_name efficientnet-b3_10x

# Batch processing
python skin_seg_inference.py /path/to/images/ --batch --model_name efficientnet-b3_10x
```

### Manual Model Loading

```python
from skin_seg_inference import SkinSegmentationModel
from PIL import Image

# Initialize model with magnification awareness
model = SkinSegmentationModel(
    model_name="efficientnet-b3_10x",
    requested_magnification="10x"
)

# Load and predict
image = Image.open("skin_sample.jpg")
pred_mask, confidence = model.predict(image)
```

## 📊 Key Features

- **🎯 Magnification Awareness**: Models explicitly specify their training magnification
- **🔍 Multi-Scale Support**: Handles images from 1x to 40x magnification
- **🖼️ Whole Slide Imaging**: Native support for WSI formats (NDPI, SVS, etc.)
- **⚡ Batch Processing**: Efficient processing of multiple images
- **📈 Comprehensive Analysis**: Generates detailed tissue statistics and visualizations
- **🎨 Rich Visualizations**: 6-panel analysis with confidence maps and class distributions

## 🔧 Model Architecture

All models use a U-Net architecture with pre-trained backbones:
- **EfficientNet**: CNN backbones optimized for efficiency and accuracy
- **GigaPath**: Vision transformer backbone for pathology applications
- **Input**: 224×224 RGB patches
- **Output**: 12-class segmentation masks with confidence scores

## 📏 Magnification Guidelines

| Magnification | Use Case | Recommended Model |
|---------------|----------|-------------------|
| 10x | General skin analysis | `efficientnet-b3_10x` |
| 20x | Detailed cellular analysis | `efficientnet-b7_10x` (if available) |
| Other | Mixed magnifications | `efficientnet-b5` (general) |

## 🔄 Migration from Legacy Models

If you were using models without magnification suffixes:
- `efficientnet-b3_unet_best.pt` → `efficientnet-b3_10x_unet_best.pt`
- `efficientnet-b7_unet_best.pt` → `efficientnet-b7_10x_unet_best.pt`

The inference script automatically handles this migration when you specify `--magnification`.

## 📦 Installation Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install huggingface-hub
pip install openslide-python  # For WSI support
```

## 🎓 Citation

If you use these models in your research, please cite:

```bibtex
@misc{skin_seg_models_2024,
  title={Skin Histopathology Segmentation Models},
  author={JoshuaFreeman},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/JoshuaFreeman/skin_seg}
}
```

## 📄 License

These models are released under the MIT License. See LICENSE for details.

## 🤝 Contributing

Found an issue or want to contribute? Please open an issue or pull request in the associated repository.

---
**🔬 Powered by state-of-the-art deep learning for advancing skin pathology research**