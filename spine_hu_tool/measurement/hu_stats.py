"""HU statistics over an ROI mask (the measurement itself -- a simple average)."""
from __future__ import annotations
import numpy as np


def compute_hu_stats(hu: np.ndarray, roi_mask: np.ndarray, spacing) -> dict:
    v = hu[roi_mask]
    if v.size == 0:
        return {"voxel_count": 0, "mean_HU": float("nan")}
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    return {
        "mean_HU": float(np.mean(v)),
        "median_HU": float(np.median(v)),
        "sd_HU": float(np.std(v)),
        "min_HU": float(np.min(v)),
        "max_HU": float(np.max(v)),
        "p05_HU": float(np.percentile(v, 5)),
        "p95_HU": float(np.percentile(v, 95)),
        "voxel_count": int(v.size),
        "volume_mm3": float(v.size * voxel_vol),
    }
