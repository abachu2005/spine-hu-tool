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
# Cylinder ROI: axis along SI, height = fraction of body SI extent (capped to
# stay inside the central, endplate-safe band); radius solved from the same
# volume-proportional target as the sphere, capped by in-plane cortical clearance.
CYLINDER_HEIGHT_FRAC = 0.50      # cylinder height as a fraction of body height

# --- Segmentation consistency gate --------------------------------------------
# Sanity bounds on each vertebra label (full mask: body + posterior elements).
SEG_MIN_BODY_VOL_MM3 = 1500.0    # reject specks (e.g. a 37-voxel mislabel)
# Ceiling for a single full vertebra label (body + posterior elements). On
# thick-slice (~3.75 mm) scans of large patients a legitimate lumbar label
# reaches ~90-100 cm3, so 90 cm3 was too tight and falsely rejected clean
# levels (e.g. an L3 at 98 cm3 next to an 87 cm3 L2 that passed). 140 cm3 keeps
# single vertebrae valid while still catching true two-level merges (~175 cm3+)
# and over-grown sacrum blobs (~200 cm3+).
SEG_MAX_BODY_VOL_MM3 = 140000.0  # reject merged/over-grown blobs
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

# --- Scout (localizer) body-habitus measurement --------------------------------
# A scout is a projection radiograph, so "air" is NOT -1000 HU: the values are
# line integrals and the background sits near -450 on the scanners seen so far.
# Every threshold is therefore expressed RELATIVE to a per-image background
# estimate and a per-image body-attenuation reference, never as an absolute HU.
SCOUT_BG_BORDER_COLS = 15        # columns at each edge used to estimate air
# The body-attenuation reference is measured PER ROW (then smoothed along z), not
# once for the whole image: a scan whose field includes the shoulders and arms has
# a far higher global peak than one that does not, and a global reference would
# make the same patient measure ~10 mm narrower on the study that includes them.
SCOUT_REF_PERCENTILE = 98.0      # per-row percentile of (value - background)
SCOUT_REF_SMOOTH_MM = 15.0       # z-smoothing of the per-row reference
# Body edge = largest contiguous run above this fraction of the body reference.
# The CT couch projects as a low, flat plateau (~8% of body attenuation in the
# lateral view), so this threshold also excludes the table without a dedicated
# table model; taking the LARGEST run additionally rejects table-rail spikes.
SCOUT_BODY_THRESHOLD_FRAC = 0.15
SCOUT_SMOOTH_MM = 5.0            # median-smoothing of the per-row width profile
SCOUT_LEVEL_BAND_FRAC = 0.60     # central fraction of a level's SI extent to average
# Divergent-beam magnification: apparent = true * SOD / (SOD + d), where d is the
# body-center offset along the beam axis. The AP and lateral views measure each
# other's offset, so the correction is solved by iterating between them.
SCOUT_MAG_ITERS = 2
SCOUT_BEAM_SIGN = 1.0            # +1: image-plane normal points away from the source
SCOUT_MAX_CENTER_OFFSET_MM = 60.0   # larger -> flag (correction becomes unreliable)
SCOUT_EDGE_MARGIN_MM = 5.0       # extent this close to the image edge -> clipped
SCOUT_MIN_PLAUSIBLE_MM = 80.0    # outside this band the row is not a torso cross
SCOUT_MAX_PLAUSIBLE_MM = 500.0   # (e.g. arms/shoulders in the projected field)
# Rows within one level should agree closely; a large spread means the outline
# jumped (typically between an arm and the torso) rather than following skin.
SCOUT_MAX_ROW_SPREAD_MM = 40.0
# At and above T2 the shoulder girdle and upper arms project over the torso, so
# the AP "width" there is a shoulder width. Anatomy, not a fitted threshold.
SCOUT_SHOULDER_GIRDLE_LEVELS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                "T1", "T2")

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
    cylinder_height_frac: float = CYLINDER_HEIGHT_FRAC

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoutParams:
    """Parameters that fully determine a scout body-habitus measurement."""
    bg_border_cols: int = SCOUT_BG_BORDER_COLS
    ref_percentile: float = SCOUT_REF_PERCENTILE
    ref_smooth_mm: float = SCOUT_REF_SMOOTH_MM
    body_threshold_frac: float = SCOUT_BODY_THRESHOLD_FRAC
    smooth_mm: float = SCOUT_SMOOTH_MM
    level_band_frac: float = SCOUT_LEVEL_BAND_FRAC
    mag_iters: int = SCOUT_MAG_ITERS
    beam_sign: float = SCOUT_BEAM_SIGN
    max_center_offset_mm: float = SCOUT_MAX_CENTER_OFFSET_MM
    edge_margin_mm: float = SCOUT_EDGE_MARGIN_MM
    min_plausible_mm: float = SCOUT_MIN_PLAUSIBLE_MM
    max_plausible_mm: float = SCOUT_MAX_PLAUSIBLE_MM
    max_row_spread_mm: float = SCOUT_MAX_ROW_SPREAD_MM

    def to_dict(self) -> dict:
        return asdict(self)
