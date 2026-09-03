"""Find and load CT scout (localizer) projections for a study.

The measurement pipeline deliberately *rejects* localizers (see
``io/series_selector._classify``) because they are useless for HU. This module
is the other half of that decision: it collects the very series the measurement
path throws away, because the scout is the only place the patient's whole body
outline exists. Spine protocols reconstruct a tight, spine-centered axial FOV
(280 mm / 246 mm on the project test data) that cuts off the flanks and the
belly, whereas the scout spans the full ~530 mm scan field.

Nothing here changes series selection; the HU path is untouched.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

AXIS_NAMES = ("x", "y", "z")


@dataclass
class ScoutImage:
    """One localizer projection, with the geometry needed to measure in mm.

    Array convention is the DICOM one for a 2D image: ``pixels[row, col]``.
    Columns run along ``row_dir`` (a patient axis: x for an AP/PA view, y for a
    lateral view) and rows run along ``col_dir`` (always the z / SI axis for a
    CT localizer).
    """
    pixels: np.ndarray                 # 2D, rescaled to the file's HU-like scale
    view: str                          # "AP" (measures LR) or "LAT" (measures AP)
    axis: int                          # patient axis index sampled along columns
    beam_axis: int                     # patient axis the x-ray beam travels along
    row_spacing_mm: float              # between rows (along col_dir, i.e. z)
    col_spacing_mm: float              # between columns (along row_dir)
    origin: tuple                      # ImagePositionPatient (x, y, z) mm
    row_dir: tuple                     # unit vector along increasing column index
    col_dir: tuple                     # unit vector along increasing row index
    sid_mm: Optional[float] = None     # DistanceSourceToDetector
    sod_mm: Optional[float] = None     # DistanceSourceToPatient (source->isocenter)
    table_height_mm: Optional[float] = None
    series_uid: str = ""
    study_uid: str = ""
    description: str = ""
    path: str = ""
    warnings: list = field(default_factory=list)

    # --- geometry -------------------------------------------------------
    def column_to_patient(self, col) -> np.ndarray:
        """Column index -> patient coordinate (mm) along `self.axis`."""
        col = np.asarray(col, dtype=float)
        return self.origin[self.axis] + self.row_dir[self.axis] * col * self.col_spacing_mm

    def row_to_z(self, row) -> np.ndarray:
        row = np.asarray(row, dtype=float)
        return self.origin[2] + self.col_dir[2] * row * self.row_spacing_mm

    def z_to_row(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        step = self.col_dir[2] * self.row_spacing_mm
        if step == 0:
            return np.zeros_like(z)
        return (z - self.origin[2]) / step

    @property
    def z_range(self) -> tuple:
        zs = self.row_to_z([0, self.pixels.shape[0] - 1])
        return float(min(zs)), float(max(zs))

    @property
    def axis_name(self) -> str:
        return AXIS_NAMES[self.axis]

    def summary(self) -> dict:
        lo, hi = self.z_range
        return {
            "view": self.view,
            "measures_axis": self.axis_name,
            "beam_axis": AXIS_NAMES[self.beam_axis],
            "shape": list(self.pixels.shape),
            "row_spacing_mm": round(self.row_spacing_mm, 4),
            "col_spacing_mm": round(self.col_spacing_mm, 4),
            "z_range_mm": [round(lo, 1), round(hi, 1)],
            "sid_mm": self.sid_mm,
            "sod_mm": self.sod_mm,
            "table_height_mm": self.table_height_mm,
            "series_uid": self.series_uid,
            "description": self.description,
            "warnings": list(self.warnings),
        }


def _as_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_scout(path: str) -> Optional[ScoutImage]:
    """Read one localizer file into a :class:`ScoutImage` (None if unusable)."""
    import pydicom
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
    except Exception:
        return None
    if arr is None or arr.ndim != 2:
        return None

    slope = _as_float(getattr(ds, "RescaleSlope", None), 1.0)
    inter = _as_float(getattr(ds, "RescaleIntercept", None), 0.0)
    pixels = arr.astype(np.float32) * slope + inter

    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    ps = getattr(ds, "PixelSpacing", None)
    if iop is None or ipp is None or ps is None or len(iop) != 6:
        return None
    row_dir = np.array([float(v) for v in iop[:3]])
    col_dir = np.array([float(v) for v in iop[3:]])
    origin = tuple(float(v) for v in ipp)

    # The patient axis sampled along columns decides what this view measures:
    # columns along x -> an AP/PA projection giving left-right width; columns
    # along y -> a lateral projection giving anterior-posterior depth.
    axis = int(np.argmax(np.abs(row_dir)))
    if axis == 2:                      # columns along z: not a standard localizer
        return None
    view = "AP" if axis == 0 else "LAT"
    # The beam travels along the image-plane normal, i.e. the remaining in-plane
    # patient axis (rows are always the SI axis on a CT localizer).
    beam_axis = 1 if axis == 0 else 0

    warnings = []
    if abs(col_dir[2]) < 0.99:
        warnings.append("scout rows are not aligned with the SI axis")

    sid = _as_float(getattr(ds, "DistanceSourceToDetector", None))
    sod = _as_float(getattr(ds, "DistanceSourceToPatient", None))
    if not sid or not sod or sod <= 0:
        warnings.append("no source/detector geometry; magnification not corrected")

    return ScoutImage(
        pixels=pixels, view=view, axis=axis, beam_axis=beam_axis,
        row_spacing_mm=float(ps[0]), col_spacing_mm=float(ps[1]),
        origin=origin, row_dir=tuple(row_dir), col_dir=tuple(col_dir),
        sid_mm=sid, sod_mm=sod,
        table_height_mm=_as_float(getattr(ds, "TableHeight", None)),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        description=str(getattr(ds, "SeriesDescription", "")),
        path=path, warnings=warnings,
    )


def find_scouts(folder: str, study_uid: Optional[str] = None,
                candidates: Optional[list] = None) -> list[ScoutImage]:
    """Localizer projections in `folder`, optionally restricted to one study.

    `candidates` may be a previously computed ``scan_series()`` result so a
    caller that already walked the folder does not pay for a second pass.
    Returns at most one image per view, preferring the largest (the scanner
    sometimes stores several); AP first, then lateral.
    """
    from ..io.series_selector import scan_series
    if candidates is None:
        candidates = scan_series(folder)

    out: list[ScoutImage] = []
    for info in candidates:
        if "LOCALIZER" not in set(info.image_type):
            continue
        if study_uid and info.study_uid != study_uid:
            continue
        for path in info.files:
            img = load_scout(path)
            if img is not None:
                out.append(img)

    best: dict[str, ScoutImage] = {}
    for img in out:
        prev = best.get(img.view)
        if prev is None or img.pixels.size > prev.pixels.size:
            best[img.view] = img
    return [best[v] for v in ("AP", "LAT") if v in best]
