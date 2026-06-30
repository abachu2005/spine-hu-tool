"""Central configuration constants for the pipeline.

All tunables live here so a measurement can be reproduced exactly from the
recorded parameter set. Values are deliberately conservative and documented.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


# --- Metal / artifact detection ------------------------------------------------
METAL_HU_THRESHOLD = 2500.0      # HU above this is treated as metal hardware
STREAK_BUFFER_MM = 10.0          # dilation around metal to capture streak/bloom

# --- ROI geometry --------------------------------------------------------------
CENTRAL_HEIGHT_FRAC = 0.60       # use only the central 60% of body height (SI)
TARGET_RADIUS_MM = 8.0           # desired sphere radius (capped by anatomy)
RADIUS_FRAC = 0.60               # radius <= frac * max safe radius at center
MIN_RADIUS_MM = 3.0              # below this, flag for review
# Cortical margin is size-proportional: margin = max(MARGIN_FLOOR_MM,
#   MARGIN_FRAC * max_safe_radius). This adapts to body size / rind thickness.
MARGIN_FLOOR_MM = 3.0
MARGIN_FRAC = 0.20
# Lowest-attenuation sphere (Westerhoff et al.): ROI volume = fraction of the
# vertebral body volume, positioned at the lowest-attenuation interior spot
# anterior to the basivertebral foramen.
ROI_VOLUME_FRAC = 0.05           # sphere volume as a fraction of body volume
ANTERIOR_FRAC = 0.60             # keep the anterior 60% of AP extent as candidates

# --- Segmentation consistency gate --------------------------------------------
# Sanity bounds on each vertebra label (full mask: body + posterior elements).
SEG_MIN_BODY_VOL_MM3 = 1500.0    # reject specks (e.g. a 37-voxel mislabel)
SEG_MAX_BODY_VOL_MM3 = 90000.0   # reject merged/over-grown blobs
SEG_MIN_CC_FRACTION = 0.55       # largest connected component / total
SEG_MAX_Z_OVERLAP_FRAC = 0.50    # SI overlap between NON-adjacent levels
SEG_MIN_STEP_MM = 8.0            # per-VERT_ORDER-step centroid spacing (SI)
SEG_MAX_STEP_MM = 55.0
SEG_INVALID_FRACTION = 0.34      # > 1/3 of levels invalid -> whole seg invalid

# --- QC thresholds -------------------------------------------------------------
SD_FLAG_HU = 150.0               # ROI standard deviation above this -> heterogeneous
RETAINED_FRACTION_FLAG = 0.95    # sphere clipped if less than this fraction kept
CORTICAL_TAIL_P95_HU = 400.0     # p95 inside ROI above this hints cortex contamination
MIN_ROI_VOLUME_MM3 = 100.0       # ROI smaller than this -> too small

# --- Literature anchoring (reporting only; not used for decisions) -------------
L1_OSTEOPOROSIS_HU = 110.0       # approx; Pickhardt et al.
L1_NORMAL_HU = 160.0

# Per-vertebra diagnostic thresholds for the lowest-attenuation 3D ROI, from
# Westerhoff et al. (2025), Table 6 (22nd / 65th percentile in women aged
# 60-69). (osteoporosis_HU, osteopenia_HU): HU < osteoporosis -> osteoporosis;
# osteoporosis <= HU < osteopenia -> osteopenia; else normal. These apply ONLY
# to the calibrated lowest_attenuation_sphere median; not to centered/2D ROIs.
VERT_THRESHOLDS_HU = {
    "T1": (124, 170), "T2": (117, 162), "T3": (110, 156), "T4": (106, 151),
    "T5": (103, 147), "T6": (99, 142), "T7": (95, 137), "T8": (93, 134),
    "T9": (95, 136), "T10": (99, 140), "T11": (96, 136), "T12": (87, 125),
    "L1": (80, 117), "L2": (73, 112), "L3": (69, 106), "L4": (70, 108),
    "L5": (79, 119),
}


@dataclass
class ROIParams:
    """Parameters that fully determine an ROI measurement."""
    central_height_frac: float = CENTRAL_HEIGHT_FRAC
    target_radius_mm: float = TARGET_RADIUS_MM
    radius_frac: float = RADIUS_FRAC
    min_radius_mm: float = MIN_RADIUS_MM
    margin_floor_mm: float = MARGIN_FLOOR_MM
    margin_frac: float = MARGIN_FRAC
    roi_volume_frac: float = ROI_VOLUME_FRAC
    anterior_frac: float = ANTERIOR_FRAC

    def to_dict(self) -> dict:
        return asdict(self)
