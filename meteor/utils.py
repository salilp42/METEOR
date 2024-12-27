"""
Utility functions for METEOR.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import yaml
import pandas as pd

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML/JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    Dict containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_results(
    results: Dict[str, Any],
    output_path: str,
    save_timeseries: bool = True
) -> None:
    """
    Save analysis results to files.
    
    Parameters
    ----------
    results : Dict
        Analysis results
    output_path : str
        Path to save results
    save_timeseries : bool
        Whether to save time series data separately
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Separate time series data if present
    stats_dict = {}
    timeseries_dict = {}
    
    for roi_name, roi_data in results.items():
        if isinstance(roi_data, dict) and 'temporal_features' in roi_data:
            # Handle time series data
            stats_dict[roi_name] = roi_data['temporal_features']
            if save_timeseries:
                timeseries_dict[roi_name] = {
                    'mean_curve': roi_data['mean_curve'],
                    'std_curve': roi_data['std_curve']
                }
        else:
            # Handle regular stats
            stats_dict[roi_name] = roi_data
    
    # Save main statistics
    df = pd.DataFrame.from_dict(stats_dict, orient="index")
    df.to_csv(output_path)
    
    # Save time series if present
    if timeseries_dict and save_timeseries:
        ts_path = os.path.splitext(output_path)[0] + "_timeseries.csv"
        ts_df = pd.DataFrame(timeseries_dict)
        ts_df.to_csv(ts_path)

def validate_inputs(
    main_path: str,
    roi_paths: List[str],
    temporal: bool = False
) -> None:
    """
    Validate input parameters.
    
    Parameters
    ----------
    main_path : str
        Path to main image
    roi_paths : List[str]
        List of ROI paths
    temporal : bool
        Whether temporal analysis is requested
    """
    # Check if files exist
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main image not found: {main_path}")
    
    for roi_p in roi_paths:
        if not os.path.exists(roi_p):
            raise FileNotFoundError(f"ROI not found: {roi_p}")
    
    # Check file extensions
    valid_exts = {'.nii', '.nii.gz', '.mha', '.nrrd', '.dcm', '.dicom'}
    main_ext = os.path.splitext(main_path)[1].lower()
    if main_ext not in valid_exts and not os.path.isdir(main_path):
        raise ValueError(f"Unsupported file format: {main_ext}")
        
    for roi_p in roi_paths:
        roi_ext = os.path.splitext(roi_p)[1].lower()
        if roi_ext not in valid_exts and not os.path.isdir(roi_p):
            raise ValueError(f"Unsupported ROI format: {roi_ext}")

class METEORError(Exception):
    """Base exception for METEOR errors."""
    pass

class DimensionMismatchError(METEORError):
    """Raised when image dimensions don't match."""
    pass

class EmptyROIError(METEORError):
    """Raised when ROI is empty."""
    pass

class OrientationMismatchError(METEORError):
    """Raised when image orientations don't match."""
    pass
