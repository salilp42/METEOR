"""
Command-line interface for METEOR.
"""

import os
import sys
import logging
import click
from typing import List, Dict, Any

from .io.image import (
    load_image, check_orientation, resample_to_reference,
    sitk_to_np, get_spacing_for_numpy
)
from .core.stats import (
    compute_basic_stats, compute_additional_stats,
    compute_entropy, compute_volume, compute_surface_area
)
from .core.timeseries import (
    extract_timeseries, compute_temporal_features, detect_motion
)
from .visualization.viewer import visualize_with_napari, quick_plot
from .utils import (
    load_config, save_results, validate_inputs,
    METEORError, DimensionMismatchError, EmptyROIError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_case(
    main_path: str,
    roi_paths: List[str],
    morph: bool = False,
    plot: bool = False,
    napari_vis: bool = False,
    temporal: bool = False,
    tr: float = None,
    motion_check: bool = False
) -> Dict[str, Dict[str, float]]:
    """Process a single case with main image and ROIs."""
    try:
        validate_inputs(main_path, roi_paths, temporal)
        
        logger.info(f"Loading main volume: {main_path}")
        main_img = load_image(main_path)
        main_arr = sitk_to_np(main_img)

        if main_arr.ndim == 3 and plot:
            quick_plot(main_arr)
        elif main_arr.ndim == 4 and plot:
            quick_plot(main_arr, time_idx=0)

        all_stats = {}
        roi_masks = []
        time_data = None

        for roi_p in roi_paths:
            roi_img = load_image(roi_p)
            if not check_orientation(main_img, roi_img):
                logger.warning(f"Orientation mismatch for ROI: {roi_p}")
            
            roi_img = resample_to_reference(roi_img, main_img, "nearest", 0.0)
            roi_arr = sitk_to_np(roi_img)

            # Create binary mask
            bin_mask = roi_arr > 0
            roi_masks.append(bin_mask)

            roi_name = os.path.basename(roi_p)
            
            if temporal and main_arr.ndim == 4:
                # Extract time series
                ts_data = extract_timeseries(main_arr, bin_mask)
                time_data = ts_data  # Save for visualization
                
                # Compute temporal features
                temp_features = compute_temporal_features(ts_data['timeseries'], tr)
                
                # Motion detection
                if motion_check:
                    motion_frames = detect_motion(ts_data['timeseries'])
                    if motion_frames:
                        logger.warning(f"Potential motion detected at frames: {motion_frames}")
                
                all_stats[roi_name] = {
                    'temporal_features': temp_features,
                    'mean_curve': ts_data['mean_curve'].tolist(),
                    'std_curve': ts_data['std_curve'].tolist()
                }
            else:
                # Extract voxels for statistics
                voxels = main_arr[bin_mask] if main_arr.ndim == 3 else main_arr[0][bin_mask]
                
                # Compute statistics
                stats = compute_basic_stats(voxels)
                stats.update(compute_additional_stats(voxels))
                stats["entropy"] = compute_entropy(voxels)

                # Get spacing in numpy order
                spacing = get_spacing_for_numpy(main_img)
                
                # Compute geometric measures
                stats["volume_mm3"] = compute_volume(bin_mask, spacing)
                surface_area = compute_surface_area(bin_mask, spacing)
                if surface_area is not None:
                    stats["surface_area_mm2"] = surface_area

                all_stats[roi_name] = stats

        if napari_vis:
            visualize_with_napari(main_arr, roi_masks, time_data)

        return all_stats
        
    except Exception as e:
        logger.error(f"Error processing case: {str(e)}")
        raise

def process_batch(config_path: str) -> List[Dict[str, Any]]:
    """Process multiple cases defined in a YAML/JSON config file."""
    config = load_config(config_path)
    
    results = []
    cases = config.get("cases", [])
    output_dir = config.get("output_folder")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, case in enumerate(cases):
        logger.info(f"Processing case {i+1}/{len(cases)}")
        try:
            stats = process_single_case(
                case["main"],
                case.get("rois", []),
                morph=case.get("morph", False),
                temporal=case.get("temporal", False),
                tr=case.get("tr", None),
                motion_check=case.get("motion_check", False)
            )
            results.append(stats)

            if output_dir:
                out_csv = os.path.join(output_dir, f"case_{i+1}_stats.csv")
                save_results(stats, out_csv)
                
        except Exception as e:
            logger.error(f"Error processing case {i+1}: {str(e)}")
            continue

    return results

@click.command()
@click.option("--input", "-i", "input_path", help="Path to main image or DICOM folder")
@click.option("--roi", "-r", "roi_paths", multiple=True, help="One or more ROI images")
@click.option("--batch", "-b", "batch_yaml", help="YAML/JSON config for batch processing")
@click.option("--morph", "-m", is_flag=True, help="Apply morphological operations")
@click.option("--plot", is_flag=True, help="Display quick plot if 3D")
@click.option("--napari", is_flag=True, help="Open napari for visualization")
@click.option("--csv", "csv_out", help="CSV path to save statistics")
@click.option("--temporal", "-t", is_flag=True, help="Extract temporal features for 4D data")
@click.option("--tr", type=float, help="Repetition time in seconds (for temporal analysis)")
@click.option("--motion-check", is_flag=True, help="Check for motion artifacts in 4D data")
def main(
    input_path: str,
    roi_paths: tuple,
    batch_yaml: str,
    morph: bool,
    plot: bool,
    napari: bool,
    csv_out: str,
    temporal: bool,
    tr: float,
    motion_check: bool
) -> None:
    """METEOR - Medical Extraction Tool for Enhanced ROI."""
    try:
        if batch_yaml:
            results = process_batch(batch_yaml)
            logger.info("Batch processing complete")
            return

        if not input_path:
            logger.error("Please provide --input or --batch")
            sys.exit(1)

        stats = process_single_case(
            input_path,
            list(roi_paths),
            morph=morph,
            plot=plot,
            napari_vis=napari,
            temporal=temporal,
            tr=tr,
            motion_check=motion_check
        )

        if csv_out:
            save_results(stats, csv_out)
            logger.info(f"Results saved to {csv_out}")
            
    except METEORError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
