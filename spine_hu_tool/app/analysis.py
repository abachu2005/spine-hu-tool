"""Shared analysis driver used by both the GUI worker and the CLI.

Handles: series selection -> load volume -> segmentation (cached) ->
per-level QC + ROI measurement. Caching is keyed by series UID so re-opening a
dataset is instant.
"""
from __future__ import annotations
import os
import hashlib
from typing import Callable, Optional

from ..io.series_selector import select_ct_series, SeriesInfo
from ..io.dicom_loader import load_series
from ..segmentation.backends import segment, resolve_seg_url
from ..config import ROIParams
from .review_state import ReviewState

ProgressCb = Optional[Callable[[str, float], None]]


def _seg_cache_dir() -> str:
    """Folder-independent cache, keyed by series UID.

    The segmentation depends only on the scan (series), not on which folder the
    physician happened to open. Storing it in a stable per-user location means
    opening a single patient folder or a parent folder full of patients reuses
    the same cached result -- no redundant re-segmentation/upload.
    """
    d = os.environ.get("SPINE_HU_CACHE_DIR") or os.path.join(
        _user_cache_root(), "spine_hu_tool", "seg")
    os.makedirs(d, exist_ok=True)
    return d


def _user_cache_root() -> str:
    """OS-appropriate per-user cache root.

    Windows -> %LOCALAPPDATA%, macOS -> ~/Library/Caches, Linux/other ->
    $XDG_CACHE_HOME or ~/.cache. Keeps the packaged app's cache where each OS
    expects it instead of always using ~/.cache.
    """
    import sys
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return base
    elif sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return xdg
    return os.path.join(os.path.expanduser("~"), ".cache")


def _safe_name(series: SeriesInfo) -> str:
    return hashlib.sha1(series.series_uid.encode()).hexdigest()[:12]


def _migrate_legacy_cache(folder: str, name: str, cache: str) -> None:
    """One-time copy of an older per-folder cache into the shared cache."""
    dest = os.path.join(cache, f"{name}_seg.nii.gz")
    if os.path.exists(dest):
        return
    import shutil
    # Older per-folder caches, then the previous always-~/.cache shared cache.
    legacy_paths = [
        os.path.join(folder, ".spine_hu_cache", f"{name}_seg.nii.gz"),
        os.path.join(folder, "..", ".spine_hu_cache", f"{name}_seg.nii.gz"),
        os.path.join(os.path.expanduser("~"), ".cache", "spine_hu_tool",
                     "seg", f"{name}_seg.nii.gz"),
    ]
    for legacy in legacy_paths:
        if os.path.abspath(legacy) != os.path.abspath(dest) and os.path.exists(legacy):
            shutil.copy2(legacy, dest)
            return


def analyze_dataset(folder: str, series: Optional[SeriesInfo] = None,
                    mode: str = "centroid_volume_sphere", only_clean: bool = True,
                    fast: Optional[bool] = None, reviewer: str = "unknown",
                    params: Optional[ROIParams] = None,
                    seg_url: Optional[str] = None, api_key: Optional[str] = None,
                    local: bool = False,
                    compute_comparison: bool = False,
                    apply_calibration: bool = True,
                    progress: ProgressCb = None) -> ReviewState:
    def report(msg, frac):
        if progress:
            progress(msg, frac)

    # Resolution policy: full-res (1.5 mm) gives the best ROI placement but
    # needs ~12 GB RAM, so default to it only when segmentation is offloaded to
    # the cloud; run laptop-local segmentation in fast (3 mm) mode by default.
    remote = (not local) and resolve_seg_url(seg_url) is not None
    if fast is None:
        fast = not remote

    # Progress model:
    #   ingest/select : 0.00 - 0.08 (determinate)
    #   segmentation  : indeterminate (frac = -1.0; UI shows a busy bar) because
    #                   TotalSegmentator gives no callback and dominates runtime
    #                   on a first/uncached run
    #   measurement   : 0.10 - 1.00 (determinate, reported per vertebral level)
    report("Selecting CT series...", 0.03)
    if series is None:
        series, _cands = select_ct_series(folder)
    if series is None:
        raise RuntimeError("No usable axial CT series found in folder.")

    report(f"Loading series ({series.n_files} slices)...", 0.08)
    volume = load_series(series.files, metadata={
        "series_uid": series.series_uid,
        "series_desc": series.description,
        "kernel": series.kernel,
        "slice_thickness": series.slice_thickness,
        "kvp": series.kvp,
        "manufacturer_model": series.manufacturer_model,
    })

    name = _safe_name(series)
    cache = _seg_cache_dir()
    _migrate_legacy_cache(folder, name, cache)
    seg_cached = os.path.exists(os.path.join(cache, f"{name}_seg.nii.gz"))
    if seg_cached:
        report("Loading cached segmentation...", 0.10)
    elif remote:
        report("Segmenting vertebrae on the cloud service "
               "(uploading scan, this can take a few minutes)...", -1.0)
    else:
        report("Segmenting vertebrae (first run also downloads the model; "
               "this can take a few minutes)...", -1.0)
    seg = segment(volume, cache, name, fast=fast, local=local,
                  seg_url=seg_url, api_key=api_key,
                  progress=lambda m, _f: report(m, -1.0))

    audit_path = os.path.join(cache, f"{name}_audit.json")
    state = ReviewState.from_volume(
        volume, seg, mode=mode, only_clean=only_clean, params=params,
        compute_comparison=compute_comparison, apply_calibration=apply_calibration,
        reviewer=reviewer, audit_path=audit_path,
        progress=lambda m, f: report(m, 0.10 + 0.90 * f))
    report("Done.", 1.0)
    return state
