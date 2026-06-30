"""Coordinate transforms and spacing-aware morphology helpers.

Array convention: [x, y, z] with spacing (sx, sy, sz) in mm.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, label,
                           distance_transform_edt)


def index_to_physical(idx, spacing, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    idx = np.asarray(idx, dtype=float)
    return idx * np.asarray(spacing) + np.asarray(origin)


def physical_to_index(phys, spacing, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    phys = np.asarray(phys, dtype=float)
    return (phys - np.asarray(origin)) / np.asarray(spacing)


def ball(radius_mm: float, spacing) -> np.ndarray:
    """Ellipsoidal structuring element of physical radius `radius_mm`."""
    sx, sy, sz = spacing
    rx = max(1, int(round(radius_mm / sx)))
    ry = max(1, int(round(radius_mm / sy)))
    rz = max(1, int(round(radius_mm / sz)))
    xx, yy, zz = np.ogrid[-rx:rx + 1, -ry:ry + 1, -rz:rz + 1]
    return ((xx * sx) ** 2 + (yy * sy) ** 2 + (zz * sz) ** 2) <= radius_mm ** 2


def dilate_mm(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    if not mask.any():
        return mask
    return binary_dilation(mask, structure=ball(radius_mm, spacing))


def erode_mm(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    if not mask.any():
        return mask
    return binary_erosion(mask, structure=ball(radius_mm, spacing))


def within_mm(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """Voxels within `radius_mm` of `mask`, via a spacing-aware distance
    transform. Far faster than dilation with a large structuring element and
    equivalent for proximity tests."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm


def largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def make_sphere(shape, center_idx, radius_mm: float, spacing) -> np.ndarray:
    """Boolean sphere of physical radius `radius_mm` centered at a voxel index."""
    sx, sy, sz = spacing
    cx, cy, cz = center_idx
    X, Y, Z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                          np.arange(shape[2]), indexing="ij")
    d2 = ((X - cx) * sx) ** 2 + ((Y - cy) * sy) ** 2 + ((Z - cz) * sz) ** 2
    return d2 <= radius_mm ** 2


def crop_bbox(mask: np.ndarray, spacing, pad_mm: float = 20.0):
    """Return a tuple of slices bounding `mask` padded by pad_mm (clamped)."""
    idx = np.argwhere(mask)
    lo = idx.min(0)
    hi = idx.max(0) + 1
    pad = np.array([int(round(pad_mm / s)) for s in spacing])
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
