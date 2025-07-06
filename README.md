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

### 1. Cellular Density Analysis (NDPI Files)

**Purpose**: Count nuclei in whole slide images for rejection monitoring in transplant samples.

**Key Features**:
- Adaptive thresholding with red channel intensity filtering
- Contour-based tissue region detection
- Parallel tile processing for large images
- Comprehensive statistical analysis
- Heatmap visualization of cellular density

**Technical Details**:
- Uses OTSU thresholding with R+B channel filtering
- Configurable kernel operations for morphological processing
- Percentile-based red intensity value analysis
- Multi-level image pyramid processing

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

# Process all NDPI files in a directory
python main.py --input-dir ../../ndpi_files --output-dir ../../results

# Process a single file with custom config
python main.py --input-file ../../ndpi_files/sample1.ndpi \
               --output-dir ../../results \
               --config ../../config.yaml

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

The analysis generates:

```
results/
├── sample1/
│   ├── overview.png              # Whole slide overview with contours
│   ├── mask_1_contour_1/
│   │   └── tile_mosaic.png      # Processed tiles for contour 1
│   ├── mask_1_contour_2/
│   │   └── tile_mosaic.png      # Processed tiles for contour 2
│   └── results.json             # Numerical results
├── sample2/
│   └── ...
├── summary.csv                  # Summary statistics for all slides
├── detailed.csv                 # Detailed per-contour results
├── analysis_report.txt          # Human-readable report
└── configuration.json           # Analysis parameters used
```

**Key Output Metrics**:
- **Total nuclei count** per slide and contour
- **Nuclei density** (nuclei per mm²)
- **Tissue area** analyzed (mm²)
- **Confidence scores** and processing statistics

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

## 🔧 Configuration Parameters

### Cellular Density Analysis

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `threshold_param` | Adaptive threshold block size | 13 | 3-31 (odd) |
| `rpb_threshold_percentile` | Red channel intensity cutoff | 50.0 | 0-100 |
| `kernel_size` | Morphological operations | [3,3] | [1,1] to [15,15] |
| `tile_size` | Analysis tile size (pixels) | 128 | 64-512 |
| `contour_level` | Image pyramid level | 6 | 0-8 |
| `min_coverage_fraction` | Min tissue per tile | 0.25 | 0.1-0.8 |

### Key Algorithms

1. **Histogram Stretching**: Enhances contrast for better nuclei detection
2. **OTSU Formula**: Automatic threshold selection for nuclei segmentation
3. **R+B Channel Filtering**: Reduces background noise using color information
4. **Adaptive Thresholding**: Handles varying illumination across slides
5. **Morphological Operations**: Cleans up segmentation artifacts

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

### Workflow 1: Transplant Rejection Monitoring
```bash
# 1. Place NDPI files from patient biopsies
mkdir patient_samples
cp *.ndpi patient_samples/

# 2. Run cellular density analysis
python backend/wsi_processing/main.py \
  --input-dir patient_samples \
  --output-dir rejection_analysis \
  --config transplant_monitoring_config.yaml

# 3. Review results in rejection_analysis/summary.csv
```

### Workflow 2: Comprehensive Skin Analysis
```bash
# 1. Run both cellular density and tissue segmentation
python backend/wsi_processing/main.py --input-dir skin_samples --output-dir analysis

# 2. Extract skin patches and run segmentation
python backend/skin_seg_inference.py skin_patches/ --batch --model_name efficientnet-b5

# 3. Combine results for comprehensive analysis
```

## 🎯 Clinical Applications

### Transplant Monitoring
- **Rejection Assessment**: Monitor cellular density changes over time
- **Treatment Response**: Track nuclei count changes following interventions
- **Comparative Analysis**: Compare pre/post treatment samples

### Dermatopathology
- **Tissue Classification**: Automated identification of skin structures
- **Cancer Detection**: Identify and quantify malignant regions
- **Research Applications**: Standardized tissue analysis for studies

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