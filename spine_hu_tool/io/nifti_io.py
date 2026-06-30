"""NIfTI read/write helpers.

We persist volumes/masks to NIfTI both for caching and as the TotalSegmentator
input. The affine encodes voxel spacing so external tools (e.g. 3D Slicer)
load the geometry correctly.
"""
from __future__ import annotations
import numpy as np
import nibabel as nib

from ..core import Volume


# DICOM/SimpleITK use LPS world coordinates; NIfTI uses RAS. Converting is just
# negating the x and y world axes.
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def _affine(spacing, origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    """RAS affine for a volume given its LPS spacing/origin/direction cosines.

    A correct affine is essential for TotalSegmentator: it reorients the volume
    to canonical space using this matrix, so an identity-but-wrong affine makes
    it segment a flipped/rotated body and can mislabel vertebral levels.
    """
    S = np.asarray(spacing, dtype=float)
    D = np.asarray(direction, dtype=float).reshape(3, 3)   # LPS, row-major
    O = np.asarray(origin, dtype=float)
    aff = np.eye(4)
    aff[:3, :3] = _LPS_TO_RAS @ (D @ np.diag(S))
    aff[:3, 3] = _LPS_TO_RAS @ O
    return aff


def save_volume_nifti(volume: Volume, path: str, dtype=np.float32) -> str:
    """Write a Volume to NIfTI. Use dtype=np.int16 for uploads -- CT HU are
    integral and within int16 range, so it's lossless but ~half the size."""
    arr = volume.hu
    if np.issubdtype(np.dtype(dtype), np.integer):
        arr = np.rint(arr)
    img = nib.Nifti1Image(arr.astype(dtype),
                          _affine(volume.spacing, volume.origin, volume.direction))
    nib.save(img, path)
    return path


def save_mask_nifti(mask: np.ndarray, spacing, path: str,
                    origin=(0.0, 0.0, 0.0)) -> str:
    img = nib.Nifti1Image(mask.astype(np.uint8), _affine(spacing, origin))
    nib.save(img, path)
    return path


def load_nifti_array(path: str) -> tuple[np.ndarray, tuple]:
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing
