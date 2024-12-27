"""
Visualization utilities using napari and matplotlib.
"""

import logging
from typing import List, Optional, Dict, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import napari
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False

logger = logging.getLogger(__name__)

def visualize_with_napari(
    img_arr: np.ndarray,
    roi_masks: List[np.ndarray],
    time_data: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Show main volume and ROIs in napari with optional time series data.
    
    Parameters
    ----------
    img_arr : np.ndarray
        3D or 4D image array
    roi_masks : List[np.ndarray]
        List of ROI masks
    time_data : Dict[str, np.ndarray], optional
        Time series data to plot alongside volume
    """
    if not HAS_NAPARI:
        logger.warning("napari not installed, skipping visualization")
        return

    viewer = napari.Viewer()
    
    if img_arr.ndim == 3:
        viewer.add_image(img_arr, name="Main Volume")
    elif img_arr.ndim == 4:
        # Rearrange from (T,Z,Y,X) to (Z,Y,X,T) for napari
        arr_for_napari = np.transpose(img_arr, (1, 2, 3, 0))
        viewer.add_image(arr_for_napari, name="Main Volume 4D")
    
    for i, rmask in enumerate(roi_masks):
        viewer.add_labels(rmask.astype(int), name=f"ROI_{i}")
    
    if time_data and HAS_MPL:
        # Create matplotlib figure for time series
        plt.figure(figsize=(10, 4))
        if 'mean_curve' in time_data:
            plt.plot(time_data['mean_curve'], label='Mean')
            if 'std_curve' in time_data:
                mean = time_data['mean_curve']
                std = time_data['std_curve']
                plt.fill_between(
                    range(len(mean)),
                    mean - std,
                    mean + std,
                    alpha=0.2
                )
        plt.title('ROI Time Series')
        plt.xlabel('Time Point')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
    
    napari.run()

def quick_plot(
    img_arr: np.ndarray,
    slice_idx: Optional[int] = None,
    time_idx: Optional[int] = None
) -> None:
    """
    Display a quick matplotlib plot of the data.
    
    Parameters
    ----------
    img_arr : np.ndarray
        3D or 4D image array
    slice_idx : int, optional
        Index of slice to show (for 3D/4D)
    time_idx : int, optional
        Index of timepoint to show (for 4D)
    """
    if not HAS_MPL:
        logger.warning("matplotlib not installed, skipping plot")
        return

    if img_arr.ndim == 4:
        if time_idx is None:
            time_idx = img_arr.shape[0] // 2
        img_arr = img_arr[time_idx]

    if img_arr.ndim != 3:
        logger.warning("Quick plot only supports 3D/4D volumes")
        return

    if slice_idx is None:
        slice_idx = img_arr.shape[0] // 2

    plt.figure(figsize=(8, 8))
    plt.imshow(img_arr[slice_idx], cmap='gray')
    plt.colorbar()
    title = f"Slice {slice_idx}"
    if time_idx is not None:
        title += f" (Time {time_idx})"
    plt.title(title)
    plt.show()
