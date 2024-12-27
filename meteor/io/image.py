"""
Image I/O operations using SimpleITK.
"""

import os
import logging
from typing import Tuple, Optional
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)

def load_image(path: str) -> sitk.Image:
    """Load medical image from file or DICOM directory."""
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series_files = reader.GetGDCMSeriesFileNames(path)
        if not series_files:
            raise FileNotFoundError(f"No DICOM series found in {path}")
        reader.SetFileNames(series_files)
        return reader.Execute()
    else:
        return sitk.ReadImage(path)

def check_orientation(main_img: sitk.Image, roi_img: sitk.Image, tolerance: float = 1e-4) -> bool:
    """Check if two images have matching orientations."""
    main_dir = main_img.GetDirection()
    roi_dir = roi_img.GetDirection()
    
    for md, rd in zip(main_dir, roi_dir):
        if abs(md - rd) > tolerance:
            logger.warning("Orientation mismatch detected")
            return False
    return True

def resample_to_reference(
    moving_img: sitk.Image,
    reference_img: sitk.Image,
    interpolation: str = "nearest",
    default_value: float = 0.0
) -> sitk.Image:
    """Resample moving image to match reference image geometry."""
    interp_map = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "cubic": sitk.sitkBSpline,
        "bspline": sitk.sitkBSpline
    }
    chosen_interp = interp_map.get(interpolation.lower(), sitk.sitkNearestNeighbor)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(chosen_interp)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(moving_img)

def sitk_to_np(img_sitk: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to NumPy array."""
    return sitk.GetArrayFromImage(img_sitk)

def np_to_sitk(arr: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Convert NumPy array to SimpleITK image, copying metadata from reference."""
    img_sitk = sitk.GetImageFromArray(arr)
    img_sitk.CopyInformation(reference)
    return img_sitk

def get_spacing_for_numpy(img_sitk: sitk.Image) -> Tuple[float, float, float]:
    """Get spacing in numpy array order (z,y,x) from SimpleITK image."""
    sp = img_sitk.GetSpacing()
    return (sp[2], sp[1], sp[0])  # Convert from (x,y,z) to (z,y,x)
