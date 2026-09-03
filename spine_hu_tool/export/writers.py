"""Result, overlay, mask, reproducibility, and audit-trail export."""
from __future__ import annotations
import os
import csv
import json
import datetime as _dt
import numpy as np

from .. import __version__, config


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def write_results_csv(results: dict, path: str) -> str:
    rows = [r.summary() for r in results.values()]
    if not rows:
        open(path, "w").close()
        return path
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})
    return path


def write_results_json(case: dict, path: str) -> str:
    from ..scout.thickness import scout_json
    payload = {
        "results": {lvl: r.summary() for lvl, r in case["results"].items()},
        "level_tags": case.get("level_tags", []),
        "mode": case.get("mode"),
        "params": case.get("params"),
        "seg_status": case.get("seg_status"),
        "seg_global_reasons": (case.get("seg_check") or {}).get("global_reasons", []),
        "calibration": case.get("calibration"),
        "scout": scout_json(case.get("scout")),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_comparison(results: dict, json_path: str, csv_path: str):
    """Per-level ROI-method comparison for the reproducibility study."""
    payload = {lvl: r.comparison for lvl, r in results.items()
               if getattr(r, "comparison", None)}
    if not payload:
        return None
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    rows = []
    for lvl, comp in payload.items():
        for method, m in comp.get("methods", {}).items():
            rows.append({"level": lvl, "method": method,
                         "median_HU": m.get("median_HU"),
                         "mean_HU": m.get("mean_HU"),
                         "radius_mm": m.get("radius_mm"),
                         "voxel_count": m.get("voxel_count"),
                         "delta_median_vs_primary": m.get("delta_median_vs_primary")})
    if rows:
        keys = ["level", "method", "median_HU", "mean_HU", "radius_mm",
                "voxel_count", "delta_median_vs_primary"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in keys})
    return json_path


def write_reproducibility(case: dict, volume_meta: dict, path: str) -> str:
    """Everything needed to regenerate the measurement bit-for-bit."""
    from ..scout.thickness import scout_json
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = None
    record = {
        "tool_version": __version__,
        "created": _now(),
        "software": {"torch": torch_v},
        "config": {k: getattr(config, k) for k in dir(config)
                   if k.isupper()},
        "params": case.get("params"),
        "mode": case.get("mode"),
        "seg_status": case.get("seg_status"),
        "seg_check": case.get("seg_check"),
        "calibration": case.get("calibration"),
        "volume": volume_meta,
        "level_tags": case.get("level_tags", []),
        "scout": scout_json(case.get("scout")),
        "measurements": {lvl: r.summary() for lvl, r in case["results"].items()},
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


class AuditTrail:
    """Append-only record of physician review actions."""

    def __init__(self, path: str):
        self.path = path
        self.entries: list[dict] = []
        if os.path.exists(path):
            try:
                self.entries = json.load(open(path))
            except Exception:
                self.entries = []

    def record(self, reviewer: str, action: str, level: str, detail: dict | None = None):
        self.entries.append({"time": _now(), "reviewer": reviewer,
                             "action": action, "level": level,
                             "detail": detail or {}})
        self.flush()

    def flush(self):
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)


def write_scout_overlays(case: dict, out_dir: str) -> list[str]:
    """Render one annotated PNG per scout view (the habitus audit trail).

    The body outline is verified from these images rather than in the review UI,
    so a reviewer can confirm the traced boundary followed the skin surface and
    excluded the CT couch.
    """
    block = case.get("scout") or {}
    scouts = block.get("_scouts") or []
    profiles = block.get("_profiles") or {}
    if not scouts:
        return []
    from ..visualization.overlays import render_scout_overlay
    paths = []
    for s in scouts:
        prof = profiles.get(s.view)
        if prof is None:
            continue
        path = os.path.join(out_dir, f"scout_{s.view}.png")
        try:
            render_scout_overlay(s, prof, block.get("levels"), path)
        except Exception:
            continue        # an overlay must never break an export
        paths.append(path)
    return paths


def export_case(case: dict, volume, out_dir: str,
                overlays: bool = True, masks: bool = True) -> dict:
    """Write CSV + JSON + reproducibility (+ overlays/masks) for a case."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    written["csv"] = write_results_csv(case["results"], os.path.join(out_dir, "measurements.csv"))
    written["json"] = write_results_json(case, os.path.join(out_dir, "measurements.json"))
    vol_meta = {"shape": list(volume.shape), "spacing_mm": list(volume.spacing),
                **{k: v for k, v in (volume.metadata or {}).items()}}
    written["reproducibility"] = write_reproducibility(
        case, vol_meta, os.path.join(out_dir, "reproducibility.json"))
    comp = write_comparison(case["results"],
                            os.path.join(out_dir, "comparison.json"),
                            os.path.join(out_dir, "comparison.csv"))
    if comp:
        written["comparison"] = comp

    if overlays:
        from ..visualization.overlays import render_roi_overlay
        odir = os.path.join(out_dir, "overlays")
        os.makedirs(odir, exist_ok=True)
        for lvl, r in case["results"].items():
            if r.crop_slices is None:
                continue
            hu_crop = volume.hu[r.crop_slices]
            render_roi_overlay(hu_crop, r, volume.spacing,
                               os.path.join(odir, f"{lvl}.png"), level=lvl)
        written["overlays"] = odir
        scout_paths = write_scout_overlays(case, odir)
        if scout_paths:
            written["scout_overlays"] = ", ".join(scout_paths)

    if masks:
        from ..io.nifti_io import save_mask_nifti
        mdir = os.path.join(out_dir, "masks")
        os.makedirs(mdir, exist_ok=True)
        for lvl, r in case["results"].items():
            if r.crop_slices is None:
                continue
            # place cropped ROI back into full-volume frame for portability
            full = np.zeros(volume.shape, dtype=np.uint8)
            full[r.crop_slices][r.roi_mask] = 1
            save_mask_nifti(full, volume.spacing,
                            os.path.join(mdir, f"{lvl}_roi.nii.gz"))
            bodyfull = np.zeros(volume.shape, dtype=np.uint8)
            bodyfull[r.crop_slices][r.body_mask] = 1
            save_mask_nifti(bodyfull, volume.spacing,
                            os.path.join(mdir, f"{lvl}_body.nii.gz"))
        written["masks"] = mdir

    # Archive the exported (reviewed/final) numbers to the cloud under the same
    # run id as the analysis, best-effort in a background thread (never raises).
    from .cloud_archive import archive_export
    archive_export(case, out_dir, written)
    return written
