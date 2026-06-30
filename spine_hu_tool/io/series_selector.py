"""Series selection (data-quality only -- no privacy/PHI handling).

Scans a folder of DICOM files, groups them by SeriesInstanceUID, and ranks the
series so the real axial CT volume is auto-selected over scouts, reformats, and
secondary captures. The choice rules were validated against the project test
data (lumbar series 10000735 and thoracic series 100008B9).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pydicom


@dataclass
class SeriesInfo:
    series_uid: str
    study_uid: str
    modality: str
    series_number: Optional[int]
    description: str
    image_type: tuple
    kernel: str
    n_files: int
    files: list                      # sorted by slice position (z)
    pixel_spacing: Optional[tuple]
    slice_thickness: Optional[float]
    rows: Optional[int]
    cols: Optional[int]
    rescale_slope: Optional[float]
    rescale_intercept: Optional[float]
    uniform_spacing: bool
    z_spacing: Optional[float]
    kvp: Optional[float] = None
    manufacturer_model: str = ""
    patient_id: str = ""
    patient_name: str = ""
    study_description: str = ""
    study_date: str = ""
    is_axial_ct: bool = False
    reject_reason: Optional[str] = None
    score: float = 0.0

    @property
    def label(self) -> str:
        """Concise, unambiguous one-line series label for a chooser row."""
        bits = [self.description or "(no desc)", f"{self.modality} {self.n_files} sl"]
        if self.slice_thickness:
            bits.append(f"{self.slice_thickness:g} mm")
        if self.kvp:
            bits.append(f"{self.kvp:g} kVp")
        if self.kernel:
            bits.append(self.kernel)
        return "  \u00b7  ".join(bits)

    @property
    def patient_label(self) -> str:
        pid = self.patient_id or "Unknown patient"
        name = self.patient_name
        return f"{pid} ({name})" if name and name.lower() != "anonymized" else pid

    @property
    def study_label(self) -> str:
        desc = self.study_description or "(study)"
        return f"{desc} \u2014 {_fmt_date(self.study_date)}" if self.study_date else desc

    @property
    def group_label(self) -> str:
        """Patient + study header used to group series in the chooser."""
        return f"{self.patient_label}  \u2014  {self.study_label}"

    @property
    def option_label(self) -> str:
        """Study-centric label for a one-per-study chooser row."""
        bits = [self.study_description or self.description or "(study)"]
        if self.study_date:
            bits.append(_fmt_date(self.study_date))
        detail = [f"{self.modality} {self.n_files} sl"]
        if self.slice_thickness:
            detail.append(f"{self.slice_thickness:g} mm")
        if self.kvp:
            detail.append(f"{self.kvp:g} kVp")
        return "  \u00b7  ".join(bits) + "   (" + ", ".join(detail) + ")"


def _to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fmt_date(yyyymmdd: str) -> str:
    s = (yyyymmdd or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _image_type(ds) -> tuple:
    it = getattr(ds, "ImageType", [])
    if isinstance(it, str):
        return (it,)
    return tuple(str(x).upper() for x in it)


def scan_series(folder: str) -> list[SeriesInfo]:
    """Walk a folder recursively and group DICOM files into SeriesInfo records."""
    groups: dict[str, list] = {}
    headers: dict[str, object] = {}
    positions: dict[str, list] = {}

    for root, _dirs, files in os.walk(folder):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            uid = getattr(ds, "SeriesInstanceUID", None)
            if uid is None:
                continue
            groups.setdefault(uid, []).append(path)
            headers.setdefault(uid, ds)
            ipp = getattr(ds, "ImagePositionPatient", None)
            z = _to_float(ipp[2]) if ipp else _to_float(getattr(ds, "InstanceNumber", None))
            positions.setdefault(uid, []).append((z, path))

    out: list[SeriesInfo] = []
    for uid, paths in groups.items():
        ds = headers[uid]
        pos = positions[uid]
        # sort files by z (fallback already handled to InstanceNumber)
        pos_sorted = sorted(pos, key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0))
        sorted_files = [p for _z, p in pos_sorted]
        zs = np.array([z for z, _p in pos_sorted if z is not None], dtype=float)
        uniform, zspace = _spacing_uniformity(zs)

        kernel = getattr(ds, "ConvolutionKernel", "")
        if isinstance(kernel, (list, tuple, pydicom.multival.MultiValue)):
            kernel = "/".join(str(k) for k in kernel)
        ps = getattr(ds, "PixelSpacing", None)
        info = SeriesInfo(
            series_uid=uid,
            study_uid=str(getattr(ds, "StudyInstanceUID", "")),
            modality=str(getattr(ds, "Modality", "")),
            series_number=_to_float(getattr(ds, "SeriesNumber", None)),
            description=str(getattr(ds, "SeriesDescription", "")),
            image_type=_image_type(ds),
            kernel=str(kernel),
            n_files=len(paths),
            files=sorted_files,
            pixel_spacing=(float(ps[0]), float(ps[1])) if ps else None,
            slice_thickness=_to_float(getattr(ds, "SliceThickness", None)),
            rows=getattr(ds, "Rows", None),
            cols=getattr(ds, "Columns", None),
            rescale_slope=_to_float(getattr(ds, "RescaleSlope", None)),
            rescale_intercept=_to_float(getattr(ds, "RescaleIntercept", None)),
            uniform_spacing=uniform,
            z_spacing=zspace,
            kvp=_to_float(getattr(ds, "KVP", None)),
            manufacturer_model=str(getattr(ds, "ManufacturerModelName", "")),
            patient_id=str(getattr(ds, "PatientID", "")),
            patient_name=str(getattr(ds, "PatientName", "")),
            study_description=str(getattr(ds, "StudyDescription", "")),
            study_date=str(getattr(ds, "StudyDate", "")),
        )
        _classify(info)
        out.append(info)

    out.sort(key=lambda s: s.score, reverse=True)
    return out


def _spacing_uniformity(zs: np.ndarray, tol: float = 0.05):
    if zs.size < 3:
        return False, None
    diffs = np.diff(np.sort(zs))
    if diffs.size == 0:
        return False, None
    med = float(np.median(diffs))
    if med == 0:
        return False, None
    uniform = bool(np.all(np.abs(diffs - med) <= max(tol, 0.02 * abs(med))))
    return uniform, med


def _classify(info: SeriesInfo) -> None:
    """Decide whether a series is a usable axial CT and assign a score."""
    it = set(info.image_type)
    reasons = []

    if info.modality != "CT":
        reasons.append("not CT")
    if "LOCALIZER" in it:
        reasons.append("localizer/scout")
    if "REFORMATTED" in it:
        reasons.append("reformatted view")
    if "SECONDARY" in it:
        reasons.append("secondary capture")
    if info.n_files < 10:
        reasons.append("too few slices")
    if info.rescale_slope is None or info.rescale_intercept is None:
        reasons.append("missing rescale")
    if not info.uniform_spacing:
        reasons.append("non-uniform slice spacing")
    if "AXIAL" not in it and "PRIMARY" not in it:
        reasons.append("not a primary axial acquisition")

    if reasons:
        info.is_axial_ct = False
        info.reject_reason = "; ".join(reasons)
        info.score = -len(reasons)
        return

    info.is_axial_ct = True
    # Prefer: standard kernel (for HU), more slices, thinner slices.
    score = 100.0
    if "STANDARD" in info.kernel.upper():
        score += 20.0
    elif "BONE" in info.kernel.upper():
        score += 5.0
    score += min(info.n_files, 400) * 0.05
    if info.slice_thickness:
        score += max(0.0, 5.0 - info.slice_thickness)  # thinner is better
    info.score = score


def select_ct_series(folder: str) -> tuple[Optional[SeriesInfo], list[SeriesInfo]]:
    """Return (best_axial_ct_series, all_candidates_ranked)."""
    candidates = scan_series(folder)
    best = next((c for c in candidates if c.is_axial_ct), None)
    return best, candidates


def group_series(candidates: list[SeriesInfo], axial_only: bool = True
                 ) -> list[tuple[str, list[SeriesInfo]]]:
    """Group candidates by (patient, study) for a chooser.

    Handles the common physician workflow of pointing at a parent folder that
    contains many patients/studies. Groups are ordered so the one holding the
    best axial-CT series comes first; series within a group are best-first.
    """
    pool = [c for c in candidates if c.is_axial_ct] if axial_only else list(candidates)
    if not pool:                                   # nothing axial -> show everything
        pool = list(candidates)
    groups: dict[tuple, list[SeriesInfo]] = {}
    order: list[tuple] = []
    for c in sorted(pool, key=lambda s: s.score, reverse=True):
        key = (c.patient_id, c.study_uid)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    return [(groups[k][0].group_label, groups[k]) for k in order]


def list_studies(candidates: list[SeriesInfo], axial_only: bool = True
                 ) -> list[tuple[str, list[SeriesInfo]]]:
    """Collapse to ONE best series per study, grouped by patient.

    A CT study is typically reconstructed with several kernels (e.g. STANDARD +
    BONE) from the *same* raw data; for HU measurement only one is appropriate,
    so the chooser should not make the physician pick among redundant
    reconstructions. This returns, per patient, the single best (highest-scored)
    series for each distinct study -- i.e. real choices like "Thoracic" vs
    "Lumbar", not kernel duplicates.
    """
    pool = [c for c in candidates if c.is_axial_ct] if axial_only else list(candidates)
    if not pool:
        pool = list(candidates)
    best_per_study: dict[tuple, SeriesInfo] = {}
    for c in sorted(pool, key=lambda s: s.score, reverse=True):
        best_per_study.setdefault((c.patient_id, c.study_uid), c)
    patients: dict[str, list[SeriesInfo]] = {}
    order: list[str] = []
    for c in sorted(best_per_study.values(), key=lambda s: s.score, reverse=True):
        if c.patient_id not in patients:
            patients[c.patient_id] = []
            order.append(c.patient_id)
        patients[c.patient_id].append(c)
    return [(patients[p][0].patient_label, patients[p]) for p in order]
