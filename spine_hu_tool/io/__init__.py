"""Input/output: DICOM loading, series selection, NIfTI helpers."""
from .series_selector import SeriesInfo, scan_series, select_ct_series
from .study_discovery import StudyRecord, discover_studies, studies_from_candidates
from .dicom_loader import load_series, load_volume_from_nifti
from .nifti_io import save_volume_nifti, load_nifti_array

__all__ = [
    "SeriesInfo", "scan_series", "select_ct_series",
    "StudyRecord", "discover_studies", "studies_from_candidates",
    "load_series", "load_volume_from_nifti",
    "save_volume_nifti", "load_nifti_array",
]
