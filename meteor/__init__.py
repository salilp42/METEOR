"""
METEOR - Medical Extraction Tool for Enhanced ROI
"""

from .io.image import (
    load_image,
    check_orientation,
    resample_to_reference,
    sitk_to_np,
    np_to_sitk,
    get_spacing_for_numpy
)

from .core.stats import (
    compute_basic_stats,
    compute_additional_stats,
    compute_entropy,
    compute_volume,
    compute_surface_area,
    dice_coefficient,
    hausdorff_distance
)

from .core.timeseries import (
    extract_timeseries,
    compute_temporal_features,
    detect_motion
)

from .visualization.viewer import (
    visualize_with_napari,
    quick_plot
)

__version__ = "0.1.0"
