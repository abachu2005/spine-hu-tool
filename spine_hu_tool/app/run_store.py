"""Persistent library of analysis runs (single + batch) with reopen support.

Every analysis -- whether a single study or one study inside a batch -- is saved
here automatically so the physician can close the app and come back to review or
finalize it later, and so a batch produces a browsable set of studies. A saved
run stores:

  - identity + status metadata (``run.json``)
  - a snapshot of the measurements (for the past-runs list, no reload needed)
  - the physician's per-level review overrides (accept/reject, ROI center,
    radius) so reopening restores the exact reviewed state

Reopening rebuilds a full :class:`~spine_hu_tool.app.review_state.ReviewState`
from the original DICOM (found via the stored file list / source folder) and the
cached segmentation (keyed by series UID), then replays the overrides. The heavy
segmentation is never re-run here -- if its cache is gone we raise a clear error
rather than silently re-segmenting.

Layout (under an app-managed data dir, override with ``SPINE_HU_RUNS_DIR``)::

    <root>/runs/<run_id>/run.json
    <root>/batches/<batch_id>.json
"""
from __future__ import annotations
import os
import sys
import json
import uuid
import datetime as _dt
from typing import Optional

_STATUS = ("pending", "ready", "reviewed", "failed")


class RunReopenError(RuntimeError):
    """Raised when a saved run cannot be rebuilt (source or cache missing)."""


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _data_root() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = (os.environ.get("XDG_DATA_HOME")
                or os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(base, "SpineHUTool")


def runs_root() -> str:
    root = os.environ.get("SPINE_HU_RUNS_DIR") or os.path.join(_data_root(), "runs")
    os.makedirs(root, exist_ok=True)
    return root


def batches_root() -> str:
    root = os.path.join(os.path.dirname(runs_root()), "batches")
    os.makedirs(root, exist_ok=True)
    return root


def new_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def new_batch_id() -> str:
    return "batch-" + new_run_id()


def study_meta_from_series(series, source_folder: str = "") -> dict:
    """Build a run-library study identity from a scanned SeriesInfo."""
    files = list(getattr(series, "files", []) or [])
    if not source_folder and files:
        source_folder = os.path.dirname(os.path.abspath(files[0]))
    return {
        "patient_id": getattr(series, "patient_id", ""),
        "patient_label": getattr(series, "patient_label", ""),
        "study_uid": getattr(series, "study_uid", ""),
        "study_description": getattr(series, "study_description", ""),
        "study_date": getattr(series, "study_date", ""),
        "series_uid": series.series_uid,
        "series_description": getattr(series, "description", ""),
        "n_files": getattr(series, "n_files", len(files)),
        "source_folder": source_folder,
        "files": files,
    }


# --- serializing a ReviewState -----------------------------------------------

def overrides_from_state(state) -> dict:
    """Per-level review edits needed to restore a reviewed state on reopen."""
    out = {}
    for lvl, r in state.results.items():
        if r.body_mask is None:            # excluded level: nothing to restore
            continue
        out[lvl] = {
            "accepted": r.accepted,
            "center_idx": [int(c) for c in r.center_idx],
            "radius_mm": float(r.radius_mm),
        }
    return out


def _measurements_snapshot(state) -> dict:
    return {lvl: r.summary() for lvl, r in state.results.items()}


def build_record(state, study_meta: dict, *, backend: str, resolution: str,
                 mode: str, reviewer: str = "physician",
                 run_id: Optional[str] = None, batch_id: Optional[str] = None,
                 status: str = "ready") -> dict:
    """Assemble a run record dict from a completed ReviewState."""
    return {
        "id": run_id or new_run_id(),
        "batch_id": batch_id,
        "created": _now(),
        "updated": _now(),
        "status": status if status in _STATUS else "ready",
        "backend": backend,
        "resolution": resolution,
        "mode": mode,
        "reviewer": reviewer,
        "study": dict(study_meta),
        "error": None,
        "measurements": _measurements_snapshot(state),
        "overrides": overrides_from_state(state),
        "export_dir": None,
    }


def failed_record(study_meta: dict, error: str, *, backend: str, resolution: str,
                  mode: str, run_id: Optional[str] = None,
                  batch_id: Optional[str] = None) -> dict:
    return {
        "id": run_id or new_run_id(),
        "batch_id": batch_id,
        "created": _now(),
        "updated": _now(),
        "status": "failed",
        "backend": backend,
        "resolution": resolution,
        "mode": mode,
        "reviewer": "physician",
        "study": dict(study_meta),
        "error": str(error),
        "measurements": {},
        "overrides": {},
        "export_dir": None,
    }


# --- CRUD --------------------------------------------------------------------

def save_run(record: dict) -> str:
    rid = record["id"]
    d = os.path.join(runs_root(), rid)
    os.makedirs(d, exist_ok=True)
    record["updated"] = _now()
    path = os.path.join(d, "run.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)          # atomic, so a crash never leaves half a file
    return path


def load_run(run_id: str) -> dict:
    path = os.path.join(runs_root(), run_id, "run.json")
    with open(path) as f:
        return json.load(f)


def run_dir(run_id: str) -> str:
    return os.path.join(runs_root(), run_id)


def list_runs() -> list:
    root = runs_root()
    out = []
    for rid in os.listdir(root):
        p = os.path.join(root, rid, "run.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
    out.sort(key=lambda r: r.get("created", ""), reverse=True)
    return out


def update_run(run_id: str, **changes) -> dict:
    rec = load_run(run_id)
    rec.update(changes)
    save_run(rec)
    return rec


def save_batch(batch: dict) -> str:
    path = os.path.join(batches_root(), f"{batch['id']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(batch, f, indent=2)
    os.replace(tmp, path)
    return path


def load_batch(batch_id: str) -> dict:
    with open(os.path.join(batches_root(), f"{batch_id}.json")) as f:
        return json.load(f)


def list_batches() -> list:
    root = batches_root()
    out = []
    for fn in os.listdir(root):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(root, fn)) as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda b: b.get("created", ""), reverse=True)
    return out


def new_batch(run_ids: list, source_folder: str = "") -> dict:
    return {
        "id": new_batch_id(),
        "created": _now(),
        "updated": _now(),
        "source_folder": source_folder,
        "run_ids": list(run_ids),
        "status": "ready",
    }


# --- reopening ---------------------------------------------------------------

def apply_overrides(state, overrides: dict) -> None:
    """Replay stored per-level review edits onto a fresh ReviewState."""
    for lvl, ov in (overrides or {}).items():
        r = state.results.get(lvl)
        if r is None or r.body_mask is None:
            continue
        center = ov.get("center_idx")
        if center is not None:
            state.set_center(lvl, tuple(int(c) for c in center), record=False)
        radius = ov.get("radius_mm")
        if radius is not None:
            state.set_radius(lvl, float(radius))
        accepted = ov.get("accepted")
        if accepted is not None:
            state.set_decision(lvl, bool(accepted))


def reopen_run(run_id: str, apply_review: bool = True):
    """Rebuild a :class:`ReviewState` for a saved run.

    Raises :class:`RunReopenError` if the source DICOM or the cached
    segmentation can no longer be found.
    """
    from ..io.dicom_loader import load_series
    from ..io.series_selector import scan_series
    from ..segmentation.totalseg_runner import load_segmentation
    from .analysis import cached_seg_path, audit_path_for
    from .review_state import ReviewState

    rec = load_run(run_id)
    st = rec.get("study", {})
    series_uid = st.get("series_uid")
    if not series_uid:
        raise RunReopenError("This run has no series identity and cannot be reopened.")

    files = [f for f in (st.get("files") or []) if os.path.exists(f)]
    if not files:
        folder = st.get("source_folder")
        if folder and os.path.isdir(folder):
            match = next((c for c in scan_series(folder)
                          if c.series_uid == series_uid), None)
            if match:
                files = match.files
    if not files:
        raise RunReopenError(
            "The original DICOM files for this study could not be found "
            f"(looked under '{st.get('source_folder') or 'unknown'}'). They may "
            "have been moved or deleted; re-open the folder to analyze again.")

    seg_path = cached_seg_path(series_uid)
    if not os.path.exists(seg_path):
        raise RunReopenError(
            "The segmentation for this study is no longer cached, so it would "
            "need to be recomputed. Re-run the analysis from the folder to "
            "regenerate it.")

    volume = load_series(files, metadata={
        "series_uid": series_uid,
        "series_desc": st.get("series_description", ""),
    })
    seg = load_segmentation(seg_path)
    state = ReviewState.from_volume(
        volume, seg, mode=rec.get("mode", "centroid_volume_sphere"),
        reviewer=rec.get("reviewer", "physician"),
        audit_path=audit_path_for(series_uid))
    if apply_review:
        apply_overrides(state, rec.get("overrides", {}))
    return state
