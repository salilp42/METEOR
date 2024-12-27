"""
Time series analysis for 4D medical imaging data.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import stats, signal

def extract_timeseries(
    img_4d: np.ndarray,
    roi_mask: np.ndarray,
    temporal_axis: int = 0
) -> Dict[str, np.ndarray]:
    """
    Extract time series from 4D data using an ROI mask.
    
    Parameters
    ----------
    img_4d : np.ndarray
        4D image array with shape (T, Z, Y, X) or (Z, Y, X, T)
    roi_mask : np.ndarray
        3D binary mask with shape (Z, Y, X)
    temporal_axis : int
        Axis representing time (0 for (T,Z,Y,X), -1 for (Z,Y,X,T))
        
    Returns
    -------
    Dict containing:
        'timeseries': Raw time series for each voxel (T, N_voxels)
        'mean_curve': Mean time series across ROI (T,)
        'std_curve': Standard deviation across ROI (T,)
    """
    if temporal_axis != 0:
        img_4d = np.moveaxis(img_4d, temporal_axis, 0)
    
    # Reshape 4D to (T, N_voxels)
    t_points = img_4d.shape[0]
    spatial_shape = img_4d.shape[1:]
    
    # Flatten spatial dimensions
    img_flat = img_4d.reshape(t_points, -1)
    mask_flat = roi_mask.reshape(-1)
    
    # Extract time series for ROI voxels
    roi_timeseries = img_flat[:, mask_flat]
    
    # Compute statistics
    mean_curve = np.mean(roi_timeseries, axis=1)
    std_curve = np.std(roi_timeseries, axis=1)
    
    return {
        'timeseries': roi_timeseries,
        'mean_curve': mean_curve,
        'std_curve': std_curve
    }

def compute_temporal_features(
    timeseries: np.ndarray,
    tr: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute temporal features from a time series.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series data (T,) or (T, N_voxels)
    tr : float, optional
        Repetition time in seconds (for frequency analysis)
        
    Returns
    -------
    Dictionary of temporal features
    """
    if timeseries.ndim > 1:
        # Use mean curve if multiple voxels
        curve = np.mean(timeseries, axis=1)
    else:
        curve = timeseries
        
    # Basic temporal stats
    features = {
        'temporal_mean': np.mean(curve),
        'temporal_std': np.std(curve),
        'temporal_snr': np.mean(curve) / np.std(curve) if np.std(curve) > 0 else 0,
        'max_change': np.max(curve) - np.min(curve),
        'peak_value': np.max(curve),
        'time_to_peak': np.argmax(curve)
    }
    
    # Temporal dynamics
    if len(curve) > 1:
        # First derivative (rate of change)
        d_curve = np.diff(curve)
        features.update({
            'mean_rate': np.mean(d_curve),
            'max_rate': np.max(d_curve),
            'min_rate': np.min(d_curve)
        })
        
        # Frequency domain analysis
        if tr is not None and len(curve) > 4:
            fs = 1/tr  # Sampling frequency
            freqs, psd = signal.welch(curve, fs=fs)
            features.update({
                'peak_frequency': freqs[np.argmax(psd)],
                'power_0_01Hz': np.sum(psd[freqs < 0.01]) if any(freqs < 0.01) else 0,
                'power_0_1Hz': np.sum(psd[freqs < 0.1]) if any(freqs < 0.1) else 0
            })
    
    return features

def detect_motion(
    timeseries: np.ndarray,
    threshold: float = 2.0
) -> List[int]:
    """
    Detect potential motion artifacts in time series.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Time series data (T, N_voxels)
    threshold : float
        Z-score threshold for motion detection
        
    Returns
    -------
    List of timepoints with potential motion
    """
    if timeseries.ndim == 1:
        timeseries = timeseries[:, np.newaxis]
        
    # Compute frame-to-frame difference
    frame_diff = np.diff(timeseries, axis=0)
    mean_diff = np.mean(frame_diff, axis=1)
    
    # Z-score of frame differences
    z_scores = stats.zscore(mean_diff)
    
    # Find timepoints exceeding threshold
    motion_frames = np.where(np.abs(z_scores) > threshold)[0]
    
    return motion_frames.tolist()
