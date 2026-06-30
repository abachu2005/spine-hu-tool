"""HU = raw * RescaleSlope + RescaleIntercept.

SimpleITK already applies this when reading a series; this helper exists for
explicit/raw-pixel paths and for testing the math against known values.
"""
from __future__ import annotations
import numpy as np


def apply_rescale(raw, slope: float, intercept: float) -> np.ndarray:
    return np.asarray(raw, dtype=np.float32) * float(slope) + float(intercept)
