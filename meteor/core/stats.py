"""
Core statistical computations for ROI analysis.
"""

import numpy as np
from scipy import stats
from typing import Dict, Union, Optional

def compute_basic_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute basic statistical measures for ROI voxel intensities."""
    if data.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan
        }
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "std": float(data.std())
    }

def compute_additional_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute advanced statistical measures including skewness and kurtosis."""
    stats_dict = {}
    if data.size == 0:
        for key in ["skew", "kurtosis", "p25", "p75"]:
            stats_dict[key] = np.nan
        return stats_dict

    stats_dict["skew"] = float(stats.skew(data))
    stats_dict["kurtosis"] = float(stats.kurtosis(data))
    stats_dict["p25"] = float(np.percentile(data, 25))
    stats_dict["p75"] = float(np.percentile(data, 75))
    return stats_dict

def compute_entropy(data: np.ndarray, nbins: int = 64) -> float:
    """Compute entropy from histogram of intensities."""
    if data.size == 0:
        return np.nan
    hist, _ = np.histogram(data, bins=nbins, density=True)
    hist = hist[hist>0]  # remove zeros
    return float(-np.sum(hist * np.log2(hist)))

def compute_volume(roi_mask: np.ndarray, spacing: tuple) -> float:
    """Compute volume in mm^3 given a binary ROI and voxel spacing."""
    vox_count = roi_mask.sum()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float(vox_count * voxel_vol)

def compute_surface_area(roi_mask: np.ndarray, spacing: tuple) -> Optional[float]:
    """Compute surface area using marching cubes algorithm."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None
        
    verts, faces, _, _ = marching_cubes(roi_mask.astype(float), level=0.5, spacing=spacing)
    area = 0.0
    for tri in faces:
        pts = verts[tri]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross_prod = np.cross(v1, v2)
        tri_area = 0.5 * np.linalg.norm(cross_prod)
        area += tri_area
    return float(area)

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks."""
    intersect = (mask1 & mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()
    denom = (size1 + size2)
    if denom == 0:
        return 0.0
    return 2.0 * intersect / denom

def hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray, spacing=(1,1,1)) -> float:
    """Compute Hausdorff distance between two binary masks."""
    from scipy.spatial.distance import cdist
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)
    if coords1.shape[0] == 0 or coords2.shape[0] == 0:
        return float('inf')
    coords1 = coords1 * np.array(spacing)
    coords2 = coords2 * np.array(spacing)
    dists = cdist(coords1, coords2, 'euclidean')
    return float(max(dists.min(axis=1).max(), dists.min(axis=0).max()))
