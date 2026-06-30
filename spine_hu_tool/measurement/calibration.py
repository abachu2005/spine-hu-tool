"""Tube-voltage and scanner-model HU calibration + threshold classification.

Westerhoff et al. (2025) showed CT trabecular attenuation depends strongly on
tube voltage (80 vs 120 kVp differ by 23%) and, less so, on scanner model
(<10%). They calibrate every measurement to a 120 kVp / reference-scanner
equivalent via linear scaling factors (their Tables 2 and 3). We apply the same
factors so our HU map onto their per-vertebra thresholds (Table 6).

All factors are linear multipliers applied to the measured HU.
"""
from __future__ import annotations
from typing import Optional

from .. import config

# Table 2: tube voltage (kVp) -> factor to 120 kVp equivalent.
KVP_FACTORS = {80: 0.773, 90: 0.808, 100: 0.918, 110: 0.935, 120: 1.000, 130: 1.015}

# Table 3 (subset): scanner model -> factor to the reference scanner
# (Siemens Somatom Definition Edge = 1.000). Unlisted models -> 1.0 (uncalibrated
# for scanner; the <10% effect is recorded as a note).
SCANNER_FACTORS = {
    "Toshiba Aquilion": 0.834, "Toshiba Aquilion LB": 0.896,
    "Toshiba Aquilion Prime": 0.917, "Siemens Somatom X.cite": 0.914,
    "Siemens Somatom Drive": 0.948, "Siemens Somatom go.Top": 0.956,
    "Siemens Perspective": 0.964, "Siemens Somatom Edge Plus": 0.967,
    "Siemens Somatom Force": 0.978, "Siemens Somatom Perspective": 0.979,
    "Siemens Somatom Definition AS": 0.996, "Siemens Somatom Definition Edge": 1.000,
    "Philips Brilliance 64": 1.014, "Siemens Somatom Definition Flash": 1.014,
    "Siemens Somatom Definition": 1.035, "GE LightSpeed VCT": 1.043,
}


def _kvp_factor(kvp: Optional[float]) -> tuple[float, str]:
    if kvp is None:
        return 1.0, "kVp unknown; not calibrated for tube voltage"
    k = int(round(kvp))
    if k in KVP_FACTORS:
        return KVP_FACTORS[k], f"{k} kVp"
    nearest = min(KVP_FACTORS, key=lambda kk: abs(kk - k))
    return KVP_FACTORS[nearest], f"{k} kVp (nearest table entry {nearest})"


def _scanner_factor(model: Optional[str]) -> tuple[float, str]:
    if not model:
        return 1.0, "scanner model unknown"
    if model in SCANNER_FACTORS:
        return SCANNER_FACTORS[model], model
    return 1.0, f"scanner '{model}' not in table; no scanner correction"


def compute_calibration(metadata: dict) -> dict:
    """Combined kVp x scanner factor for a volume's acquisition metadata."""
    kvp = metadata.get("kvp")
    model = metadata.get("manufacturer_model")
    kf, knote = _kvp_factor(kvp)
    sf, snote = _scanner_factor(model)
    factor = kf * sf
    return {
        "kvp": kvp, "manufacturer_model": model,
        "kvp_factor": round(kf, 4), "scanner_factor": round(sf, 4),
        "factor": round(factor, 4),
        "calibrated": kvp is not None,
        "notes": [knote, snote],
        "reference": "120 kVp / Siemens Somatom Definition Edge (Westerhoff 2025)",
    }


def reference_context(level: str, hu: Optional[float]) -> Optional[str]:
    """Soft, non-diagnostic literature anchor for context only.

    The tool's job is to *measure* trabecular HU reproducibly; it does not assign
    a diagnostic class (thresholds are method-specific and clinically validated
    later). For the one level with a widely-cited opportunistic-CT anchor (L1,
    Pickhardt et al.), we surface where the value falls relative to those soft
    landmarks -- explicitly labeled as context, never a decision.
    """
    if level != "L1" or not isinstance(hu, (int, float)):
        return None
    if hu < config.L1_OSTEOPOROSIS_HU:
        band = f"below the L1 osteoporosis anchor (~{config.L1_OSTEOPOROSIS_HU:.0f} HU)"
    elif hu < config.L1_NORMAL_HU:
        band = (f"between the L1 osteoporosis (~{config.L1_OSTEOPOROSIS_HU:.0f}) "
                f"and normal (~{config.L1_NORMAL_HU:.0f} HU) anchors")
    else:
        band = f"at/above the L1 normal anchor (~{config.L1_NORMAL_HU:.0f} HU)"
    return f"context only (Pickhardt L1): {band}"


def classify(level: str, calibrated_hu: Optional[float]) -> Optional[str]:
    """Classify a calibrated lowest-attenuation median HU per Westerhoff Table 6.

    Returns 'osteoporosis' | 'osteopenia' | 'normal', or None if the level has
    no published threshold / HU is missing.
    """
    if calibrated_hu is None:
        return None
    th = config.VERT_THRESHOLDS_HU.get(level)
    if th is None:
        return None
    osteoporosis_hu, osteopenia_hu = th
    if calibrated_hu < osteoporosis_hu:
        return "osteoporosis"
    if calibrated_hu < osteopenia_hu:
        return "osteopenia"
    return "normal"
