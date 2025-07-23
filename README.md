# CellularDensity: Comprehensive Histopathology Analysis Platform

A medical image analysis platform for **assessing cellular density for rejection monitoring in face transplant samples**. The platform provides both nuclei counting for cellular density analysis and advanced skin tissue segmentation capabilities.

## 🎯 Overview

This platform offers two main analysis workflows:

1. **Cellular Density Analysis** - Nuclei detection and counting in NDPI whole slide images for rejection monitoring
2. **Tissue Segmentation Analysis** - 12-class skin histopathology segmentation with specialized tissue masks

## 🏗️ Architecture

```
CellularDensity/
├── backend/                     # Python analysis backend
│   ├── app.py                  # Flask web server
│   ├── segmentation_service.py # Image processing services
│   ├── skin_seg_inference.py   # Skin segmentation inference
│   ├── wsi_processing/         # WSI analysis framework
│   └── utils_*.py             # Analysis utilities
├── frontend/                   # React web interface
└── docker-compose.yaml        # Container orchestration
```

## 🔬 Analysis Capabilities

### 1. Enhanced Cellular Density Analysis (NDPI Files)

**Purpose**: Count nuclei in whole slide images for rejection monitoring in transplant samples with advanced tissue filtering capabilities.

**Key Features**:
- Adaptive thresholding with red channel intensity filtering
- **NEW: Tissue-specific cell counting** with 3 filtering modes
- **NEW: Inference tile subdivision** for improved accuracy
- **NEW: Segmentation-based background masking** (smooth BKG labels)
- Contour-based tissue region detection
- Parallel tile processing for large images
- Comprehensive statistical analysis with tissue filtering metrics
- Enhanced tile mosaics with tissue overlay visualizations

**Advanced Tissue Filtering**:
- **Mode 0 (None)**: Count all tissue areas (default behavior)
- **Mode 1 (Exclude Structural)**: Exclude GLD+KER+HYP from counting (gray overlay)
- **Mode 2 (Dermis+Hypodermis Only)**: Count only in RET+PAP+EPI tissue (blue overlay)

**Technical Details**:
- Uses OTSU thresholding with R+B channel filtering
- Segmentation-driven tissue classification for filtering
- Inference tile subdivision with configurable sizes
- Mask-based nuclei filtering using bitwise operations
- Smooth background handling from BKG segmentation labels
- Multi-level image pyramid processing with tissue awareness

### 2. Skin Tissue Segmentation

**Purpose**: Segment skin histopathology into 12 tissue classes with specialized clinical masks.

**Key Tissue Masks**:
- 🔴 **Epidermis (EPI)** - Class 6
- 🔵 **Dermis (RET+PAP)** - Classes 4,5 (Reticular + Papillary)
- 🟢 **Structural (GLD+KER+HYP)** - Classes 0,7,3 (Gland + Keratin + Hypodermis)

**12-Class Segmentation**:
| Class | Tissue Type | Color | Description |
|-------|-------------|-------|-------------|
| 0 | GLD | Purple | Glandular structures |
| 1 | INF | Dark Pink | Inflammatory regions |
| 2 | FOL | Bright Pink | Follicular structures |
| 3 | HYP | Light Pink | Hypodermis/subcutaneous |
| 4 | RET | Dark Red | Reticular dermis |
| 5 | PAP | Pink | Papillary dermis |
| 6 | EPI | Dark Purple | Epidermis |
| 7 | KER | Light Pink | Keratinizing regions |
| 8 | BKG | Black | Background |
| 9 | BCC | Cyan | Basal cell carcinoma |
| 10 | SCC | Light Green | Squamous cell carcinoma |
| 11 | IEC | Light Red | Invasive epithelial carcinoma |

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision segmentation-models-pytorch
pip install albumentations pillow numpy matplotlib
pip install openslide-python  # For NDPI support
pip install huggingface_hub    # For model downloads
pip install flask flask-cors   # For web server
```

### Environment Setup

1. **Create `.env` file**:
```bash
# Backend configuration
BACKEND_PORT=8000
HF_TOKEN=your_huggingface_token_here  # Optional for model downloads
```

2. **Docker setup** (recommended):
```bash
docker-compose up --build
```

## 📁 Processing NDPI Files (Cellular Density Analysis)

### Step 1: Organize Your Data

Create the following directory structure:
```
your_project/
├── ndpi_files/          # Place your .ndpi files here
│   ├── sample1.ndpi
│   ├── sample2.ndpi
│   └── sample3.ndpi
└── results/             # Results will be generated here
```

### Step 2: Configure Analysis Parameters

Create a configuration file `config.yaml`:
```yaml
# Cellular density analysis configuration
input_dir: "./ndpi_files"
output_dir: "./results"
file_pattern: "*.ndpi"

nuclei_detection:
  threshold_param: 13        # Adaptive threshold parameter
  rpb_threshold_percentile: 50.0  # Red channel percentile cutoff
  kernel_size: [3, 3]       # Morphological kernel size
  dilation_iterations: 2    # Number of dilation iterations

tiling:
  tile_size: 128           # Tile size in pixels
  contour_level: 6         # Pyramid level for contour detection
  min_coverage_fraction: 0.25  # Minimum tissue coverage per tile

masking:
  top_n_contours: 5        # Number of largest contours to analyze
  min_contour_area: 1000   # Minimum contour area in pixels
```

### Step 3: Run Analysis

#### Option A: WSI Processing Framework (Recommended)
```bash
cd backend/wsi_processing

# Basic processing (no tissue filtering)
python main.py --input-dir ../../ndpi_files --output-dir ../../results

# NEW: Process with dermis+hypodermis tissue filtering
python main.py \
  --input-file "../../ndpi_files/sample.ndpi" \
  --output-dir ../../results_dermis_hypodermis \
  --tissue-filter dermis-hypodermis-only \
  --segmentation-model efficientnet-b3

# NEW: Exclude structural tissue from counting
python main.py \
  --input-file "../../ndpi_files/sample.ndpi" \
  --output-dir ../../results_exclude_structural \
  --tissue-filter exclude-structural \
  --segmentation-model efficientnet-b5

# Process with custom config and tissue filtering
python main.py --input-file ../../ndpi_files/sample1.ndpi \
               --output-dir ../../results \
               --config ../../config.yaml \
               --tissue-filter dermis-hypodermis-only \
               --segmentation-model efficientnet-b3

# See all options
python main.py --help
```

#### Option B: Direct Python Usage
```bash
cd backend
python example_usage.py
```

#### Option C: Web Interface
```bash
# Start the backend server
cd backend
python app.py

# In another terminal, start frontend (if using web interface)
cd frontend
npm start

# Open http://localhost:3000 in browser
```

### Step 4: Understanding Results

The analysis generates enhanced outputs with tissue filtering capabilities:

```
results/
├── sample1/
│   ├── overview.png              # Whole slide overview with contours
│   ├── mask_1_contour_1/
│   │   ├── tile_mosaic.png      # Enhanced tiles with tissue overlays
│   │   └── inference_tiles/     # NEW: Subdivided inference tiles
│   │       ├── tile_0/
│   │       │   ├── inf_tile_0_nuclei_15_filtered.png
│   │       │   ├── inf_tile_1_nuclei_8_filtered.png
│   │       │   └── inference_mosaic.png
│   │       └── tile_1/
│   │           └── ...
│   ├── mask_1_contour_2/
│   │   └── ...
│   └── results.json             # Enhanced results with tissue filtering
├── sample2/
│   └── ...
├── summary.csv                  # Summary statistics for all slides
├── detailed.csv                 # Detailed per-contour results
├── analysis_report.txt          # Human-readable report
└── configuration.json           # Analysis parameters used
```

**Enhanced Output Metrics**:
- **Total nuclei count** per slide and contour (tissue-filtered)
- **Nuclei density** (nuclei per mm²) in allowed tissue areas
- **Tissue area** analyzed (mm²) with filtering breakdown
- **Tissue filtering mode** and description
- **Inference tiles statistics** (total/valid counts)
- **Confidence scores** and processing statistics

**NEW: Tile Mosaic Visualizations**:
- **Gray overlay (30% alpha)**: Shows excluded structural tissue (Mode 1)
- **Blue overlay (30% alpha)**: Shows included dermis+hypodermis (Mode 2)
- **Black background**: Smooth BKG segmentation labels instead of blue
- **Green overlays**: Detected nuclei locations
- **Individual inference tiles**: Saved with nuclei counts in filenames

## 🎨 Skin Tissue Segmentation

### Quick Usage

```bash
cd backend

# List available models
python skin_seg_inference.py --list_models

# Analyze single image with auto-downloaded model
python skin_seg_inference.py path/to/skin_image.jpg --model_name efficientnet-b3

# Analyze batch of images
python skin_seg_inference.py path/to/images/ --batch --model_name efficientnet-b5

# Use local model
python skin_seg_inference.py image.jpg --model_path ./my_model.pt --backbone resnet50
```

### Available Models

Models automatically download from [HuggingFace: JoshuaFreeman/skin_seg](https://huggingface.co/JoshuaFreeman/skin_seg):

- `efficientnet-b3` (~53MB) - Fast, good accuracy
- `efficientnet-b5` (~126MB) - Higher accuracy, slower
- `gigapath` - Specialized histopathology foundation model
- Custom models you upload to the repository

### Output

Segmentation analysis generates:
- **6-panel comprehensive visualization** with tissue masks
- **Individual binary masks** for each tissue type
- **Quantitative statistics** (percentages, areas)
- **Batch processing CSV** for multiple images

## 🏋️ Model Training

### Training Custom Segmentation Models

The platform includes a comprehensive training framework for fine-tuning models on your own skin histopathology data.

#### Quick Start Training

```bash
cd backend

# Show available model architectures and their defaults
python train_segmentation.py --show_defaults

# Train with GigaPath backbone (optimized for histopathology)
python train_segmentation.py --root /path/to/dataset/data --backbone gigapath_vitl

# Train with EfficientNet-B5 (good balance of speed/accuracy)
python train_segmentation.py --root /path/to/dataset/data --backbone efficientnet-b5

# Train with DINOv2 (self-supervised vision transformer)
python train_segmentation.py --root /path/to/dataset/data --backbone vit_base_patch14_dinov2

# Resume training from checkpoint
python train_segmentation.py --root /path/to/dataset/data --resume_from_checkpoint auto
```

#### Dataset Preparation

1. **Directory Structure**:
```
dataset/
├── data/
│   ├── Images/          # RGB images (TIFF/PNG)
│   │   ├── case1_tile001.tif
│   │   └── case1_tile002.tif
│   └── Masks/           # RGB masks with color-coded labels
│       ├── case1_tile001.png
│       └── case1_tile002.png
├── train_files.txt      # List of training samples
└── validation_files.txt # List of validation samples
```

2. **File Lists Format**:
```
# train_files.txt (no extensions, just stems)
case1_tile001
case1_tile002
case2_tile001
```

3. **Mask Color Coding** (RGB values):
```python
# 12-class color map for masks
(108,   0, 115): 0  # GLD - Gland
(145,   1, 122): 1  # INF - Inflammation
(216,  47, 148): 2  # FOL - Follicle
(254, 246, 242): 3  # HYP - Hypodermis
(181,   9, 130): 4  # RET - Reticular
(236,  85, 157): 5  # PAP - Papillary
( 73,   0, 106): 6  # EPI - Epidermis
(248, 123, 168): 7  # KER - Keratin
(  0,   0,   0): 8  # BKG - Background
(127, 255, 255): 9  # BCC - Basal Cell Carcinoma
(127, 255, 142): 10 # SCC - Squamous Cell Carcinoma
(255, 127, 127): 11 # IEC - Inflammatory/Epithelial Cells
```

#### Available Backbone Models

**Histopathology Foundation Models**:
- `gigapath_vitl` - Prov-GigaPath ViT-Large (1.3B histopathology tiles)

**Self-supervised Vision Transformers**:
- `vit_small_patch14_dinov2` - DINOv2 ViT-Small/14
- `vit_base_patch14_dinov2` - DINOv2 ViT-Base/14 
- `vit_large_patch14_dinov2` - DINOv2 ViT-Large/14
- `vit_giant_patch14_dinov2` - DINOv2 ViT-Giant/14

**CNN Models (ImageNet pretrained)**:
- `resnet34`, `resnet50`, `resnet101` - ResNet architectures
- `efficientnet-b3`, `efficientnet-b5`, `efficientnet-b7` - EfficientNet family
- `resnext50_32x4d` - ResNeXt architecture
- `densenet121` - DenseNet architecture
- `mobilenet_v2` - Lightweight mobile architecture

#### Training Configuration

Model-specific defaults are automatically applied from `model_configs.json`:

```bash
# Override specific parameters
python train_segmentation.py --root /path/to/data \
  --backbone efficientnet-b5 \
  --lr 1e-4 \
  --bs 32 \
  --epochs 40 \
  --freeze_encoder_epochs 2
```

#### Advanced Training Options

```bash
# Save checkpoints during training
python train_segmentation.py --root /path/to/data \
  --save_checkpoints \
  --checkpoint_interval 5

# Resume from specific checkpoint
python train_segmentation.py --root /path/to/data \
  --resume_from_checkpoint ./checkpoints/checkpoint_epoch_015.pt

# Custom file extensions
python train_segmentation.py --root /path/to/data \
  --img_ext .png \
  --mask_ext .tif
```

#### Training Output

Training generates:
- **Best model weights**: `{backbone}_unet_best.pt`
- **Training metrics**: Logged to Weights & Biases
- **Checkpoints**: Optional periodic saves
- **Visualization**: Loss curves and sample predictions in W&B

#### Model Deployment

After training, use your model for inference:
```bash
# Use your trained model
python skin_seg_inference.py image.jpg --model_path ./efficientnet-b5_unet_best.pt

# Upload to HuggingFace for easy sharing
# 1. Create a model repository on HuggingFace
# 2. Upload your .pt file
# 3. Others can use: --model_name your-username/your-model
```

## 🔧 Configuration Parameters

### Enhanced Cellular Density Analysis

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `threshold_param` | Adaptive threshold block size | 13 | 3-31 (odd) |
| `rpb_threshold_percentile` | Red channel intensity cutoff | 50.0 | 0-100 |
| `kernel_size` | Morphological operations | [3,3] | [1,1] to [15,15] |
| `tile_size` | Analysis tile size (pixels) | 128 | 64-512 |
| `contour_level` | Image pyramid level | 6 | 0-8 |
| `min_coverage_fraction` | Min tissue per tile | 0.25 | 0.1-0.8 |
| **`tissue_filter`** | **Tissue filtering mode** | **none** | **none, exclude-structural, dermis-hypodermis-only** |
| **`segmentation_model`** | **HF model for segmentation** | **None** | **efficientnet-b3, efficientnet-b5, gigapath** |
| **`inference_tile_size`** | **Inference subdivision size** | **256** | **128-512** |

### NEW: Tissue Filtering Modes

| Mode | Parameter | Included Tissues | Excluded Tissues | Overlay Color |
|------|-----------|------------------|------------------|---------------|
| 0 | `none` | All tissue types | Background only | None |
| 1 | `exclude-structural` | INF, FOL, RET, PAP, EPI, BCC, SCC, IEC | GLD, KER, HYP, BKG | Gray (30% alpha) |
| 2 | `dermis-hypodermis-only` | RET, PAP, EPI | All others | Blue (30% alpha) |

### Key Algorithms

1. **Histogram Stretching**: Enhances contrast for better nuclei detection
2. **OTSU Formula**: Automatic threshold selection for nuclei segmentation
3. **R+B Channel Filtering**: Reduces background noise using color information
4. **Adaptive Thresholding**: Handles varying illumination across slides
5. **Morphological Operations**: Cleans up segmentation artifacts
6. **NEW: Segmentation-Based Filtering**: Uses tissue segmentation for targeted counting
7. **NEW: Mask Intersection**: Bitwise operations for nuclei-tissue filtering
8. **NEW: Inference Tile Subdivision**: Subdivides tiles for improved processing accuracy

## 📊 Performance Considerations

### Hardware Requirements
- **RAM**: 8GB minimum, 16GB+ recommended for large slides
- **Storage**: ~2-5GB per processed slide for full output
- **GPU**: Optional for segmentation (speeds up inference 5-10x)

### Processing Times
- **NDPI Analysis**: 5-15 minutes per slide (depending on size and complexity)
- **Segmentation**: 1-3 seconds per 224x224 patch
- **Batch Processing**: Fully parallelized for maximum efficiency

### Optimization Tips
1. **Adjust tile size** based on available RAM
2. **Reduce contour count** for faster processing
3. **Use GPU** for segmentation tasks
4. **Process multiple files** in parallel for batch jobs

## 🧪 Example Workflows

### Workflow 1: Transplant Rejection Monitoring with Tissue Filtering
```bash
# 1. Place NDPI files from patient biopsies
mkdir patient_samples
cp *.ndpi patient_samples/

# 2. Run cellular density analysis with dermis+hypodermis filtering
python backend/wsi_processing/main.py \
  --input-dir patient_samples \
  --output-dir rejection_analysis_filtered \
  --tissue-filter dermis-hypodermis-only \
  --segmentation-model efficientnet-b3 \
  --config transplant_monitoring_config.yaml

# 3. Compare with unfiltered analysis
python backend/wsi_processing/main.py \
  --input-dir patient_samples \
  --output-dir rejection_analysis_baseline \
  --tissue-filter none

# 4. Review results in rejection_analysis_*/summary.csv
```

### Workflow 2: Comprehensive Skin Analysis with Multiple Filtering Modes
```bash
# 1. Run analysis with all three filtering modes
python backend/wsi_processing/main.py \
  --input-file "skin_sample.ndpi" \
  --output-dir analysis_no_filter \
  --tissue-filter none \
  --segmentation-model efficientnet-b5

python backend/wsi_processing/main.py \
  --input-file "skin_sample.ndpi" \
  --output-dir analysis_exclude_structural \
  --tissue-filter exclude-structural \
  --segmentation-model efficientnet-b5

python backend/wsi_processing/main.py \
  --input-file "skin_sample.ndpi" \
  --output-dir analysis_dermis_hypodermis \
  --tissue-filter dermis-hypodermis-only \
  --segmentation-model efficientnet-b5

# 2. Compare nuclei density results across filtering modes
# 3. Analyze inference tiles for detailed cell distribution
```

### Workflow 3: NDPI Example File Processing
```bash
# Process the example NDPI file with dermis+hypodermis filtering
cd /tmp/CellularDensity/backend

python wsi_processing/main.py \
  --input-file "/tmp/CellularDensity/04_skin D23-031729 B1-1 - 2024-02-22 15.24.47_Skin 3A.ndpi" \
  --output-dir "/tmp/CellularDensity/results_dermis_hypodermis" \
  --tissue-filter dermis-hypodermis-only \
  --segmentation-model efficientnet-b3 \
  --tile-size 128 \
  --min-coverage-fraction 0.5 \
  --log-level INFO
```

## 🎯 Clinical Applications

### Enhanced Transplant Monitoring
- **Targeted Rejection Assessment**: Monitor cellular density in specific tissue layers (dermis/hypodermis)
- **Tissue-Specific Analysis**: Focus on clinically relevant tissue types while excluding artifacts
- **Treatment Response**: Track nuclei count changes in filtered tissue areas following interventions
- **Comparative Analysis**: Compare pre/post treatment samples with consistent tissue filtering
- **Improved Accuracy**: Eliminate structural artifacts (glands, keratin) that may confound results

### Advanced Dermatopathology
- **Precision Tissue Classification**: Automated identification with tissue-specific counting
- **Cancer Detection**: Enhanced detection in dermis/hypodermis while excluding background structures
- **Research Applications**: Standardized tissue analysis with reproducible filtering protocols
- **Multi-Modal Analysis**: Combine segmentation and cell counting for comprehensive assessment

## 🎯 NEW: Advanced Tissue Filtering Guide

### Understanding Tissue Filtering Modes

**Mode 0 (None) - Default Behavior**
```bash
--tissue-filter none
```
- Counts nuclei in all tissue areas (excludes background only)
- No visual overlays applied
- Fastest processing (no segmentation required)
- Use for: General analysis, baseline comparisons

**Mode 1 (Exclude Structural) - Gray Overlay**
```bash
--tissue-filter exclude-structural --segmentation-model efficientnet-b3
```
- Excludes: GLD (glands), KER (keratin), HYP (hypodermis)
- Includes: INF, FOL, RET, PAP, EPI, BCC, SCC, IEC
- Gray overlay (30% alpha) shows excluded areas
- Use for: Focusing on active tissue, excluding structural artifacts

**Mode 2 (Dermis+Hypodermis Only) - Blue Overlay**
```bash
--tissue-filter dermis-hypodermis-only --segmentation-model efficientnet-b3
```
- Includes ONLY: RET (reticular), PAP (papillary), EPI (epidermis)
- Excludes: All other tissue types
- Blue overlay (30% alpha) shows included areas
- Use for: Transplant monitoring, dermal-specific analysis

### Choosing the Right Segmentation Model

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `efficientnet-b3` | ~53MB | Fast | Good | General analysis, quick results |
| `efficientnet-b5` | ~126MB | Medium | Better | High-accuracy requirements |
| `gigapath` | Variable | Slow | Best | Research, maximum accuracy |

### Output Interpretation

**Enhanced Statistics**:
```json
{
  "total_nuclei_count": 1247,
  "nuclei_density_per_mm2": 892.3,
  "tissue_filter_mode": "DERMIS_HYPODERMIS_ONLY",
  "tissue_filter_description": "Counting only in dermis+hypodermis (RET+PAP+EPI) tissue",
  "inference_tiles_total": 48,
  "inference_tiles_valid": 32
}
```

**Visual Outputs**:
- **Main tile mosaic**: Shows tissue overlays and nuclei detection
- **Inference tiles**: Individual subdivided tiles with nuclei counts
- **Inference mosaics**: Reconstructed view of subdivided processing

## 🐛 Troubleshooting

### Common Issues

1. **NDPI files not processing**:
   - Ensure OpenSlide is properly installed
   - Check file permissions and corruption
   - Verify sufficient disk space

2. **Segmentation model download fails**:
   - Check internet connection
   - Install huggingface_hub: `pip install huggingface_hub`
   - Try manual model download

3. **Out of memory errors**:
   - Reduce tile_size in configuration
   - Close other applications
   - Consider processing files individually

4. **Poor nuclei detection**:
   - Adjust `threshold_param` (try 11, 15, 17)
   - Modify `rpb_threshold_percentile` (try 40-60)
   - Check image quality and staining

5. **NEW: Tissue filtering issues**:
   - **No segmentation model**: Ensure `--segmentation-model` is specified for filtering
   - **Poor tissue segmentation**: Try different models (efficientnet-b5 for better accuracy)
   - **No overlays visible**: Check if tissue filtering mode is set correctly
   - **Low nuclei counts**: Verify filtering mode matches your analysis goals

6. **NEW: Inference tile problems**:
   - **Empty inference tiles**: Reduce `min_coverage_fraction` or `inference_tile_size`
   - **Too many tiles**: Increase `inference_tile_size` for fewer subdivisions
   - **Memory issues**: Reduce `inference_tile_size` or process fewer tiles

### Getting Help

1. **Check logs**: Enable debug logging with `--log-level DEBUG`
2. **Validate configuration**: Use example configs as starting points
3. **Test with small samples**: Start with single files before batch processing
4. **Review intermediate outputs**: Check tile mosaics and contour overlays

## 📚 Technical References

### Image Processing Techniques
- **Adaptive Thresholding**: Handles varying illumination
- **Morphological Operations**: Shape-based filtering
- **Color Space Analysis**: R+B channel nuclei enhancement
- **Multi-scale Processing**: Pyramid-based analysis

### Machine Learning Models
- **Segmentation Models PyTorch**: Framework for model architectures
- **EfficientNet**: CNN backbone for segmentation
- **Foundation Models**: Domain-specific pretrained models

### File Format Support
- **NDPI**: Hamamatsu whole slide images
- **Standard Images**: PNG, JPEG, TIFF for segmentation
- **Export Formats**: CSV, JSON, PNG visualizations

---

## 📄 License

This project is released under the Apache 2.0 License.

## 🤝 Contributing

For questions, issues, or contributions, please open an issue in this repository.