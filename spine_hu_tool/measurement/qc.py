"""Per-ROI quality-control flags.

Geometry is the primary cortical-avoidance guarantee; the HU-based checks here
are a secondary, scanner-aware sanity net (e.g. a high-HU tail hints the ROI
clipped cortex). Emits pass / review / fail.
"""
from __future__ import annotations
import numpy as np

from .. import config
from ..geometry.coords import make_sphere, within_mm


def compute_qc(hu, body, roi_info, spacing, stats, params) -> dict:
    roi = roi_info["roi_mask"]
    warnings = list(roi_info.get("warnings", []))
    fail = False

    # Cortex clipping: compare the final ROI against its *intended* shape. For
    # the clipped-sphere mode the intended shape is the sphere already clipped
    # to the endplate-safe central band (band_sphere_vox); anything lost beyond
    # that is the cortex cutting in -- which is what we want to flag.
    retained = 1.0
    if np.isfinite(roi_info.get("radius_mm", float("nan"))) and roi_info["radius_mm"] > 0:
        intended = roi_info.get("band_sphere_vox")
        if intended is None:
            sphere = make_sphere(body.shape, roi_info["center_idx"],
                                 roi_info["radius_mm"], spacing)
            intended = int(sphere.sum())
        if intended > 0:
            retained = float(roi.sum() / intended)
        if retained < config.RETAINED_FRACTION_FLAG:
            warnings.append(f"ROI clipped by cortex (retained {retained:.2f})")

    # Metal proximity
    metal = hu > config.METAL_HU_THRESHOLD
    near_metal = False
    if metal.any():
        near_metal = bool((within_mm(metal, config.STREAK_BUFFER_MM, spacing) & roi).any())
        if near_metal:
            warnings.append("ROI near metal artifact")
            fail = True

    # Heterogeneity
    sd = stats.get("sd_HU", 0.0)
    if sd > config.SD_FLAG_HU:
        warnings.append(f"high ROI heterogeneity (SD {sd:.0f})")

    # Cortical-contamination hint (secondary signal)
    if stats.get("p95_HU", 0.0) > config.CORTICAL_TAIL_P95_HU:
        warnings.append(f"high-HU tail (p95 {stats['p95_HU']:.0f}); possible cortex contamination")

    # Size
    if stats.get("volume_mm3", 0.0) < config.MIN_ROI_VOLUME_MM3:
        warnings.append("ROI too small")
        fail = True
    if roi_info.get("radius_mm", 1.0) is not None and \
            np.isfinite(roi_info.get("radius_mm", float("nan"))) and \
            roi_info["radius_mm"] < params.min_radius_mm:
        warnings.append("safe radius below floor")
        fail = True

    status = "fail" if fail else ("review" if warnings else "pass")
    return {
        "qc_status": status,
        "warnings": warnings,
        "retained_fraction": round(retained, 3),
        "near_metal": near_metal,
    }
