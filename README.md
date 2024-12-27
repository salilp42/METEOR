# METEOR (Medical Extraction Tool for Enhanced ROI)

A Python-based toolkit for quantitative analysis of medical images, specializing in ROI-based time series extraction from 4D datasets. METEOR provides comprehensive analysis capabilities for:
- Dynamic imaging (fMRI, DCE-MRI, DSC-MRI)
- Static volumetric analysis (T1w, T2w)
- Multi-ROI statistical computations

## Technical Overview

### Core Architecture
- **Memory Management**: Efficient handling of large 4D datasets using NumPy memory mapping
- **Processing Engine**: Multi-threaded computation for parallel time series extraction
- **I/O Backend**: SimpleITK-based robust medical image handling with orientation preservation
- **Extensible Design**: Modular architecture supporting custom analysis pipelines

### Time Series Analysis

#### Signal Extraction
```python
{
    "timeseries": ndarray[T, N],  # T timepoints, N voxels
    "mean_curve": ndarray[T],     # ROI-averaged temporal curve
    "std_curve": ndarray[T],      # Temporal standard deviation
    "voxel_indices": ndarray[N,3] # Spatial coordinates of extracted voxels
}
```

#### Temporal Features
```python
{
    "basic_temporal": {
        "mean": float,            # Mean signal intensity
        "std": float,            # Signal standard deviation
        "snr": float,            # Signal-to-noise ratio
        "cov": float            # Coefficient of variation
    },
    "dynamic_metrics": {
        "peak_value": float,     # Maximum signal intensity
        "time_to_peak": int,     # Frame number of peak
        "wash_in_rate": float,   # Initial uptake slope
        "wash_out_rate": float,  # Post-peak slope
        "auc": float            # Area under curve
    },
    "frequency_domain": {
        "power_spectrum": ndarray,  # Frequency components
        "dominant_freq": float,    # Highest power frequency
        "spectral_entropy": float  # Frequency distribution entropy
    }
}
```

#### Motion Analysis
```python
{
    "framewise_displacement": ndarray[T-1],  # Frame-to-frame movement
    "outlier_frames": List[int],            # Detected motion timepoints
    "displacement_stats": {
        "mean_fd": float,                   # Mean framewise displacement
        "num_outliers": int,                # Number of motion events
        "max_displacement": float           # Maximum movement
    }
}
```

### Statistical Analysis

#### Basic Statistics
- **First-order**: mean, median, std, min, max, percentiles
- **Distribution**: kurtosis, skewness, entropy
- **Spatial**: volume (mm³), surface area (mm²), compactness

#### Advanced Metrics
- **Texture Analysis**: Integration with PyRadiomics for feature extraction
- **Shape Analysis**: 3D morphological characteristics
- **Intensity Distributions**: Histogram analysis and PDF fitting

## Installation

### Prerequisites
- Python >= 3.8
- C++ compiler (for SimpleITK)
- CUDA toolkit (optional, for GPU acceleration)

```bash
git clone https://github.com/salilp42/METEOR.git
cd METEOR
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Basic ROI Analysis
```bash
meteor --input t1w.nii.gz \
       --roi tumor_mask.nii.gz \
       --output-format json \
       --compute-shape \
       --csv stats.csv
```

#### Time Series Analysis
```bash
meteor --input fmri.nii.gz \
       --roi activation_mask.nii.gz \
       --temporal \
       --tr 2.0 \
       --motion-check \
       --frequency-analysis \
       --outlier-threshold 0.5 \
       --csv timeseries_stats.csv
```

#### Advanced Options
```bash
meteor --input dce_mri.nii.gz \
       --roi tumor_mask.nii.gz \
       --temporal \
       --tr 3.5 \
       --aif-mask aif_mask.nii.gz \
       --compute-ktrans \
       --denoising gaussian \
       --threads 4 \
       --output-dir ./results
```

### Python API

#### Time Series Processing
```python
from meteor import load_image, extract_timeseries, compute_temporal_features
from meteor.utils import setup_logging

# Configure logging
setup_logging(level="INFO", log_file="analysis.log")

# Load 4D image and ROI mask
main_img = load_image(
    "fmri.nii.gz",
    ensure_4d=True,           # Verify 4D structure
    check_orientation=True    # Ensure correct orientation
)
roi_img = load_image(
    "mask.nii.gz",
    ensure_binary=True,      # Verify binary mask
    resample_to=main_img     # Match main image space
)

# Extract time series with advanced options
ts_data = extract_timeseries(
    main_img,
    roi_img,
    temporal_axis=0,         # Time is first dimension
    mask_threshold=0.5,      # For probabilistic ROIs
    normalize=True           # Z-score normalization
)

# Compute comprehensive temporal features
features = compute_temporal_features(
    ts_data['timeseries'],
    tr=2.0,                 # Repetition time
    frequency_analysis=True, # Include frequency domain
    motion_detection=True,   # Check for motion
    detrend=True            # Remove linear trends
)

# Access specific metrics
mean_curve = ts_data['mean_curve']
temporal_stats = features['temporal_stats']
motion_metrics = features['motion_metrics']
```

#### Static Analysis Pipeline
```python
from meteor import (
    load_image,
    compute_basic_stats,
    compute_shape_metrics,
    extract_radiomics
)

# Load 3D image and ROI
img = load_image("t1w.nii.gz")
roi = load_image("tumor.nii.gz")

# Comprehensive analysis pipeline
results = {
    # Basic statistics
    "stats": compute_basic_stats(
        img,
        roi,
        percentiles=[1, 5, 25, 50, 75, 95, 99]
    ),
    
    # Shape characteristics
    "shape": compute_shape_metrics(
        roi,
        spacing=img.GetSpacing(),
        compute_mesh=True  # For surface analysis
    ),
    
    # Radiomics features
    "radiomics": extract_radiomics(
        img,
        roi,
        features=['firstorder', 'shape', 'glcm'],
        force_2d=False
    )
}
```

## Configuration

### Batch Processing (config.yaml)
```yaml
output_folder: "results/"
global_params:
  threads: 4
  normalize: true
  motion_correction: true
  
cases:
  - main: "sub01/fmri.nii.gz"
    rois: 
      - path: "sub01/gm_mask.nii.gz"
        label: "gray_matter"
        threshold: 0.5
      - path: "sub01/wm_mask.nii.gz"
        label: "white_matter"
        threshold: 0.5
    temporal:
      enabled: true
      tr: 2.0
      frequency_analysis: true
      detrend: true
    motion:
      check: true
      threshold: 0.5
      
  - main: "sub02/dce_mri.nii.gz"
    rois: 
      - path: "sub02/tumor.nii.gz"
    temporal:
      enabled: true
      tr: 3.5
    pharmacokinetic:
      compute_ktrans: true
      aif_mask: "sub02/aif.nii.gz"
```

## Output Formats

### Time Series JSON Structure
```json
{
  "temporal_features": {
    "basic": {
      "mean": float,
      "std": float,
      "snr": float,
      "cov": float
    },
    "dynamic": {
      "peak_value": float,
      "time_to_peak": int,
      "wash_in_rate": float,
      "wash_out_rate": float,
      "auc": float
    },
    "frequency": {
      "dominant_frequency": float,
      "power_distribution": [...],
      "spectral_entropy": float
    }
  },
  "motion_metrics": {
    "framewise_displacement": [...],
    "outlier_frames": [...],
    "summary_stats": {
      "mean_fd": float,
      "max_fd": float,
      "num_outliers": int
    }
  }
}
```

### Static Analysis CSV Format
```csv
roi_name,mean,median,std,min,max,p25,p75,volume_mm3,surface_area_mm2,compactness
roi_1,val,val,val,val,val,val,val,val,val,val
```

## Dependencies

### Core Requirements
- SimpleITK >= 2.1.0: Medical image I/O and processing
- NumPy >= 1.21.0: Numerical computations
- SciPy >= 1.7.0: Signal processing
- Pandas >= 1.3.0: Data management
- PyYAML >= 6.0.0: Configuration handling
- Click >= 8.0.0: CLI interface

### Optional Enhancements
- napari >= 0.4.12: Advanced visualization
- matplotlib >= 3.4.0: Plotting capabilities
- scikit-image >= 0.18.0: Additional image processing
- torch >= 1.9.0: Deep learning integration
- pyradiomics >= 3.0.0: Radiomics feature extraction

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License

## Citation

If you use METEOR in your research, please cite:
```bibtex
@software{meteor2024,
  title={METEOR: Medical Extraction Tool for Enhanced ROI},
  author={Salil Patel},
  year={2024},
  url={https://github.com/salilp42/METEOR}
}
