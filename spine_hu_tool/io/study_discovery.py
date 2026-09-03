"""Study-centric discovery on top of the series scanner.

The physician may point the tool at any of a wide range of folder layouts:

  - a single study folder (one patient, one study, several series/kernels)
  - a parent folder holding many patients / studies (batch)
  - deeply nested exports (patient/study/series/*.dcm) or flat dumps of files
  - PACS exports with a ``DICOMDIR`` index (the referenced image files are still
    found by the recursive scan, so no special parsing is required)

:func:`discover_studies` groups the scanned series into one record per DICOM
study (``StudyInstanceUID``), each carrying the best axial CT series to analyze,
all sibling series, and a stable ``source_folder`` so a run can be reopened
later. It reuses :mod:`spine_hu_tool.io.series_selector` for the actual DICOM
classification/scoring, so behavior stays consistent with single-study opening.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

from .series_selector import SeriesInfo, scan_series


@dataclass
class StudyRecord:
    patient_id: str
    patient_label: str
    study_uid: str
    study_label: str
    study_description: str
    study_date: str
    best_series: SeriesInfo            # the series to analyze (best axial CT)
    all_series: list = field(default_factory=list)   # every series in the study
    source_folder: str = ""            # common ancestor dir of the study's files
    has_axial: bool = False

    @property
    def n_series(self) -> int:
        return len(self.all_series)

    @property
    def option_label(self) -> str:
        return self.best_series.option_label

    def as_meta(self) -> dict:
        """JSON-serializable identity for the run library (enough to reopen)."""
        s = self.best_series
        return {
            "patient_id": self.patient_id,
            "patient_label": self.patient_label,
            "study_uid": self.study_uid,
            "study_description": self.study_description,
            "study_date": self.study_date,
            "series_uid": s.series_uid,
            "series_description": s.description,
            "n_files": s.n_files,
            "source_folder": self.source_folder,
            "files": list(s.files),
        }


def _common_folder(files: list) -> str:
    if not files:
        return ""
    abs_files = [os.path.abspath(f) for f in files]
    if len(abs_files) == 1:
        return os.path.dirname(abs_files[0])
    try:
        # commonpath of a set of files is their shared directory.
        return os.path.commonpath(abs_files)
    except ValueError:  # e.g. mixed drives on Windows
        return os.path.dirname(abs_files[0])


def studies_from_candidates(candidates: list, axial_only: bool = True
                            ) -> list:
    """Build one :class:`StudyRecord` per study from scanned series.

    Studies are ordered best-first (by the best axial CT score). When
    ``axial_only`` and at least one axial CT exists, only studies containing an
    axial CT are returned; otherwise every study is returned so nothing silently
    disappears from a batch.
    """
    axial = [c for c in candidates if c.is_axial_ct]
    pool = axial if (axial_only and axial) else list(candidates)

    best: dict = {}
    order: list = []
    for c in sorted(pool, key=lambda s: s.score, reverse=True):
        key = (c.patient_id, c.study_uid)
        if key not in best:
            best[key] = c
            order.append(key)

    all_series: dict = {}
    for c in candidates:
        key = (c.patient_id, c.study_uid)
        if key in best:
            all_series.setdefault(key, []).append(c)

    records = []
    for key in order:
        b = best[key]
        series_list = sorted(all_series.get(key, [b]),
                             key=lambda s: s.score, reverse=True)
        records.append(StudyRecord(
            patient_id=b.patient_id,
            patient_label=b.patient_label,
            study_uid=b.study_uid,
            study_label=b.study_label,
            study_description=b.study_description,
            study_date=b.study_date,
            best_series=b,
            all_series=series_list,
            source_folder=_common_folder(b.files),
            has_axial=b.is_axial_ct,
        ))
    return records


def discover_studies(root: str, axial_only: bool = True) -> list:
    """Recursively scan ``root`` and return study records (best-first)."""
    return studies_from_candidates(scan_series(root), axial_only=axial_only)
