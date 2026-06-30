"""Core data types shared across the pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Volume:
    """A CT volume in native spacing.

    hu       : float32 array [x, y, z] in Hounsfield Units
    spacing  : (sx, sy, sz) millimeters, matching the array axes
    origin   : physical origin (x0, y0, z0) in mm (optional)
    metadata : free-form dict of DICOM-derived info
    """
    hu: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: dict = field(default_factory=dict)
    # SimpleITK/DICOM direction cosines (LPS), row-major 3x3. Identity is a safe
    # default for axis-aligned volumes; the DICOM loader fills in the real value
    # so we can write a correctly oriented NIfTI for TotalSegmentator.
    direction: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.hu.shape

    @property
    def voxel_volume_mm3(self) -> float:
        return float(self.spacing[0] * self.spacing[1] * self.spacing[2])


@dataclass
class LocalAxes:
    """Per-vertebra local coordinate frame (unit column vectors in voxel space,
    scaled to physical mm consistency is handled by callers)."""
    centroid: np.ndarray            # (3,) voxel index of body centroid
    lr: np.ndarray                  # (3,) left-right axis
    ap: np.ndarray                  # (3,) anterior-posterior axis
    si: np.ndarray                  # (3,) superior-inferior axis


@dataclass
class ROIResult:
    """Output of an ROI placement for one vertebral level."""
    level: str
    mode: str
    roi_mask: np.ndarray
    center_idx: tuple[int, int, int]
    radius_mm: float
    max_safe_radius_mm: float
    margin_mm: float
    full_vox: int
    body_vox: int
    stats: dict = field(default_factory=dict)
    qc: dict = field(default_factory=dict)
    # geometry retained for visualization / review (in crop coordinates)
    crop_slices: Optional[tuple] = None
    body_mask: Optional[np.ndarray] = None
    inner_mask: Optional[np.ndarray] = None
    accepted: Optional[bool] = None        # physician decision (None = unreviewed)
    comparison: Optional[dict] = None      # per-ROI-method HU (reproducibility study)

    def summary(self) -> dict:
        d = {
            "level": self.level,
            "mode": self.mode,
            "center_idx": list(self.center_idx),
            "radius_mm": round(self.radius_mm, 2),
            "max_safe_radius_mm": round(self.max_safe_radius_mm, 2),
            "margin_mm": round(self.margin_mm, 2),
            "full_vox": int(self.full_vox),
            "body_vox": int(self.body_vox),
        }
        d.update({k: (round(v, 2) if isinstance(v, float) else v)
                  for k, v in self.stats.items()})
        d.update(self.qc)
        d["accepted"] = self.accepted        # physician decision (None = unreviewed)
        return d
