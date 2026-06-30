"""Distance-transform utilities for cortical avoidance and ROI geometry."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..geometry.local_axes import project_onto_axis


def body_distance(body: np.ndarray, spacing) -> np.ndarray:
    """Spacing-aware distance (mm) from every interior voxel to the boundary.

    This conforms to the actual irregular cortical rind -- it is not a
    spherical assumption.
    """
    return distance_transform_edt(body, sampling=spacing)


def proportional_margin(max_safe_mm: float, margin_floor_mm: float,
                        margin_frac: float) -> float:
    """Size-proportional cortical clearance (adapts to body size)."""
    return max(margin_floor_mm, margin_frac * max_safe_mm)


def make_inner(body: np.ndarray, spacing, margin_mm: float):
    dist = body_distance(body, spacing)
    return dist >= margin_mm, dist


def central_height_mask(body: np.ndarray, axes, spacing, frac: float) -> np.ndarray:
    """Mask of voxels within the central `frac` of body height along the SI axis."""
    idx = np.argwhere(body)
    if idx.size == 0:
        return body.copy()
    si = project_onto_axis(idx, axes.si, axes.centroid, spacing)
    lo, hi = si.min(), si.max()
    half = (1 - frac) / 2 * (hi - lo)
    keep = (si >= lo + half) & (si <= hi - half)
    out = np.zeros_like(body)
    sel = idx[keep]
    out[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return out


def anterior_mask(body: np.ndarray, axes, spacing, frac: float) -> np.ndarray:
    """Mask of the anterior `frac` of the body along the AP axis.

    The AP axis is sign-corrected to point posteriorly, so smaller projection =
    more anterior. Restricting candidate ROI centers here keeps them anterior to
    the basivertebral foramen (posterior-central), as in Westerhoff et al.
    """
    idx = np.argwhere(body)
    if idx.size == 0:
        return body.copy()
    ap = project_onto_axis(idx, axes.ap, axes.centroid, spacing)
    lo, hi = ap.min(), ap.max()
    keep = ap <= lo + frac * (hi - lo)
    out = np.zeros_like(body)
    sel = idx[keep]
    out[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return out
