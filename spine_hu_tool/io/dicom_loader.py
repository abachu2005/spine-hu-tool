"""Load a chosen DICOM series into a native-spacing Volume.

Uses SimpleITK for robust series reading, then converts to the project's
[x, y, z] array convention. HU is read on the native (non-interpolated) grid;
SimpleITK applies the DICOM RescaleSlope/Intercept so values are already HU.
"""
from __future__ import annotations
import numpy as np
import SimpleITK as sitk

from ..core import Volume


def _sitk_to_volume(img: sitk.Image, metadata: dict) -> Volume:
    # GetArrayFromImage returns [z, y, x]; transpose to [x, y, z].
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    hu = np.transpose(arr, (2, 1, 0))
    sx, sy, sz = img.GetSpacing()           # (x, y, z) order
    ox, oy, oz = img.GetOrigin()
    direction = tuple(float(d) for d in img.GetDirection())   # LPS, row-major 3x3
    return Volume(hu=hu, spacing=(float(sx), float(sy), float(sz)),
                  origin=(float(ox), float(oy), float(oz)),
                  direction=direction, metadata=metadata)


def load_series(files: list[str] | str, metadata: dict | None = None) -> Volume:
    """Load a DICOM series.

    `files` may be an explicit ordered list of file paths (e.g. from
    SeriesInfo.files) or a directory containing exactly one series.
    """
    reader = sitk.ImageSeriesReader()
    if isinstance(files, str):
        ids = reader.GetGDCMSeriesIDs(files)
        names = reader.GetGDCMSeriesFileNames(files, ids[0]) if ids \
            else reader.GetGDCMSeriesFileNames(files)
    else:
        names = list(files)
    reader.SetFileNames(names)
    img = reader.Execute()
    return _sitk_to_volume(img, metadata or {})


def load_volume_from_nifti(path: str, metadata: dict | None = None) -> Volume:
    """Load a Volume from a NIfTI file (used for cached test data)."""
    import nibabel as nib
    img = nib.load(path)
    hu = np.asarray(img.dataobj).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms)
    return Volume(hu=hu, spacing=spacing, metadata=metadata or {})
