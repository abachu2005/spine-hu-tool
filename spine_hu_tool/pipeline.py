"""End-to-end orchestration: volume + segmentation -> per-level HU measurements.

Each level is processed in a cropped bounding box for speed. The heavy ML
(segmentation) is done upstream; everything here is deterministic.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from .core import Volume, ROIResult
from . import config
from .config import ROIParams
from .geometry.coords import crop_bbox
from .roi.body_isolation import isolate_body
from .roi.modes import place_roi, COMPARISON_MODES
from .roi.distance import proportional_margin, body_distance
from .measurement.hu_stats import compute_hu_stats
from .measurement.qc import compute_qc
from .measurement.level_qc import tag_levels
from .measurement.calibration import compute_calibration, reference_context
from .segmentation.totalseg_runner import label_id_for, vertebra_labels
from .segmentation.consistency import check_segmentation


def _excluded_result(level: str, mode: str, reasons: list[str],
                     full_vox: int = 0) -> ROIResult:
    """A measurement-free result for a level we deliberately drop (no HU)."""
    return ROIResult(
        level=level, mode=mode, roi_mask=None, center_idx=(0, 0, 0),
        radius_mm=0.0, max_safe_radius_mm=0.0, margin_mm=0.0,
        full_vox=int(full_vox), body_vox=0, stats={},
        qc={"qc_status": "excluded", "exclusion_reason": "; ".join(reasons),
            "warnings": list(reasons), "near_metal": False,
            "retained_fraction": 0.0, "auto_excluded": True},
        crop_slices=None, body_mask=None, inner_mask=None,
    )


def _compute_comparison(H, body, spacing, params, primary_mode: str) -> dict:
    """Measure every comparison ROI method on the same body for the variance
    study, with HU deltas relative to the primary method."""
    from .roi.modes import place_roi as _place
    rows = {}
    for m in COMPARISON_MODES:
        try:
            info = _place(body, spacing, params, m, hu=H)
            s = compute_hu_stats(H, info["roi_mask"], spacing)
        except Exception as exc:  # a comparator must never break the measurement
            rows[m] = {"error": str(exc)}
            continue
        rows[m] = {"median_HU": s.get("median_HU"), "mean_HU": s.get("mean_HU"),
                   "radius_mm": round(float(info["radius_mm"]), 2),
                   "voxel_count": s.get("voxel_count"),
                   "volume_mm3": s.get("volume_mm3")}
    base = rows.get(primary_mode, {})
    base_med = base.get("median_HU")
    for m, r in rows.items():
        if isinstance(r.get("median_HU"), (int, float)) and base_med is not None:
            r["delta_median_vs_primary"] = round(r["median_HU"] - base_med, 1)
    return {"primary_mode": primary_mode, "methods": rows}


def measure_level(volume: Volume, seg: np.ndarray, level: str,
                  params: Optional[ROIParams] = None,
                  mode: str = "centroid_volume_sphere",
                  compute_comparison: bool = False) -> Optional[ROIResult]:
    params = params or ROIParams()
    hu, spacing = volume.hu, volume.spacing
    try:
        lab = label_id_for(level)
    except KeyError:
        return None
    full_f = seg == lab
    if full_f.sum() < 100:
        return None

    sl = crop_bbox(full_f, spacing)
    H = hu[sl]
    full = full_f[sl]

    body, body_flags = isolate_body(full, spacing)
    info = place_roi(body, spacing, params, mode, hu=H)
    stats = compute_hu_stats(H, info["roi_mask"], spacing)

    # --- exclusion: drop levels that can't yield a trustworthy ROI rather than
    #     reporting a garbage HU number ----------------------------------------
    # NOTE: a "partial" (FOV-clipped) body is NOT a hard exclusion. Because the
    # ROI is sized to clear every surface (including the scan boundary), the
    # sphere still lands in intact bone; we measure it but flag it for review
    # since the centroid can be biased away from the clipped end.
    reasons = []
    if body_flags.get("too_small") or body_flags.get("not_isolable"):
        reasons.append("vertebral body could not be isolated")
    r_mm = info.get("radius_mm", float("nan"))
    if not np.isfinite(r_mm) or r_mm < params.min_radius_mm:
        reasons.append(f"ROI radius {r_mm:.1f}mm below floor {params.min_radius_mm}mm")
    if stats.get("volume_mm3", 0.0) < config.MIN_ROI_VOLUME_MM3:
        reasons.append(f"ROI volume {stats.get('volume_mm3', 0.0):.0f}mm3 below minimum")

    # inner compartment retained for visualization (one distance transform
    # serves both the margin and the inner mask)
    body_dist = body_distance(body, spacing)
    margin = proportional_margin(float(body_dist.max()),
                                 params.margin_floor_mm, params.margin_frac)
    inner = body_dist >= margin

    qc = compute_qc(H, body, info, spacing, stats, params)
    persistent_warnings = list(body_flags.get("warnings", []))
    qc["warnings"] = persistent_warnings + qc["warnings"]
    if reasons:
        # Keep the ROI and raw measurement available for physician correction.
        # Reporting inclusion is a separate decision; these concerns default it
        # to excluded without turning the result into an uneditable placeholder.
        qc["qc_status"] = "fail"
        qc["auto_excluded"] = True
        qc["exclusion_reason"] = "; ".join(reasons)
        for reason in reasons:
            if reason not in qc["warnings"]:
                qc["warnings"].append(reason)
            if reason not in persistent_warnings:
                persistent_warnings.append(reason)
    if body_flags.get("partial"):
        qc["truncated"] = True
        partial_warning = (
            "review: vertebra clipped by scan field of view; verify ROI placement")
        qc["warnings"].append(partial_warning)
        persistent_warnings.append(partial_warning)
        if qc["qc_status"] == "pass":
            qc["qc_status"] = "review"
    if qc["qc_status"] == "fail":
        qc.setdefault("auto_excluded", True)
        qc.setdefault("exclusion_reason", "; ".join(qc.get("warnings", []))
                      or "failed quality control")
    if persistent_warnings:
        qc["persistent_warnings"] = persistent_warnings

    res = ROIResult(
        level=level, mode=info["mode"], roi_mask=info["roi_mask"],
        center_idx=info["center_idx"], radius_mm=info["radius_mm"],
        max_safe_radius_mm=info["max_safe_radius_mm"], margin_mm=info["margin_mm"],
        full_vox=int(full.sum()), body_vox=int(body.sum()),
        stats=stats, qc=qc, crop_slices=sl, body_mask=body, inner_mask=inner,
    )
    if compute_comparison:
        res.comparison = _compute_comparison(H, body, spacing, params, info["mode"])
    return res


def process_case(volume: Volume, seg: np.ndarray,
                 levels: Optional[list[str]] = None,
                 params: Optional[ROIParams] = None,
                 mode: str = "centroid_volume_sphere",
                 only_clean: bool = True,
                 compute_comparison: bool = False,
                 apply_calibration: bool = True,
                 progress=None) -> dict:
    """Tag levels, then measure the requested (or all clean) levels.

    `progress(msg, frac)` (frac in [0,1]) is called as each level completes so a
    UI can show accurate, incremental progress.
    """
    params = params or ROIParams()
    if progress:
        progress("Screening segmentation validity...", 0.0)
    seg_check = check_segmentation(seg, volume.spacing)

    if progress:
        progress("Tagging levels for metal/artifact...", 0.0)
    tags = tag_levels(volume.hu, seg, volume.spacing)
    tag_by_level = {t["level"]: t for t in tags}

    # When no explicit level list is given we ALWAYS walk every vertebra present
    # in the scan so the review list is complete.
    explicit_levels = levels is not None
    if not explicit_levels:
        levels = [lvl for _id, lvl in vertebra_labels(seg)]

    seg_levels = seg_check.get("levels", {})
    results: dict[str, ROIResult] = {}
    n = max(1, len(levels))
    for i, lvl in enumerate(levels):
        if progress:
            progress(f"Measuring {lvl} ({i + 1}/{n})...", (i + 1) / n)
        status = tag_by_level.get(lvl, {}).get("status")
        sv = seg_levels.get(lvl)
        default_exclusion_reasons = []
        if sv is not None and not sv["valid"]:
            default_exclusion_reasons.extend(
                "segmentation invalid: " + r for r in sv["reasons"])
        if status == "excluded":
            default_exclusion_reasons.append(
                "instrumented level: metal hardware in vertebra")

        measurement_error = None
        try:
            res = measure_level(volume, seg, lvl, params, mode,
                                compute_comparison=compute_comparison)
        except Exception as exc:
            if not default_exclusion_reasons:
                raise
            res = None
            measurement_error = str(exc)
        if res is None:
            reasons = default_exclusion_reasons or [
                "level could not produce an editable ROI measurement"]
            if measurement_error:
                reasons = reasons + [f"measurement unavailable: {measurement_error}"]
            res = _excluded_result(
                lvl, mode, reasons,
                full_vox=(sv or {}).get(
                    "voxels", tag_by_level.get(lvl, {}).get("voxels", 0)))
        else:
            if default_exclusion_reasons:
                res.qc["auto_excluded"] = True
                prior = res.qc.get("exclusion_reason")
                all_reasons = ([prior] if prior else []) + default_exclusion_reasons
                res.qc["exclusion_reason"] = "; ".join(all_reasons)
                warns = res.qc.setdefault("warnings", [])
                persistent = res.qc.setdefault("persistent_warnings", [])
                for reason in default_exclusion_reasons:
                    if reason not in warns:
                        warns.append(reason)
                    if reason not in persistent:
                        persistent.append(reason)
            # near / adjacent to instrumented hardware: streak/bloom can bias the
            # number, so surface it but flag for review instead of passing clean.
            if (res is not None and only_clean and status == "review"
                    and res.qc.get("qc_status") in ("pass", None)):
                res.qc["qc_status"] = "review"
                warning = "review: near an instrumented level (streak/bloom risk)"
                res.qc.setdefault("warnings", []).append(warning)
                res.qc.setdefault("persistent_warnings", []).append(warning)
        if res is not None:
            res.qc.setdefault("level_status", status)
            results[lvl] = res

    # --- HU calibration (no diagnostic classification; see _annotate_calibration)
    if apply_calibration:
        cal = compute_calibration(volume.metadata or {})
    else:
        cal = {"factor": 1.0, "calibrated": False,
               "notes": ["calibration disabled"], "kvp": None}
    _annotate_calibration(results, cal, mode)

    return {"level_tags": tags, "results": results, "mode": mode,
            "params": params.to_dict(),
            "seg_status": seg_check.get("seg_status"),
            "seg_check": seg_check, "calibration": cal}


def _annotate_calibration(results: dict, cal: dict, mode: str) -> None:
    """Add calibrated HU and a soft, non-diagnostic literature context note.

    The tool reports a reproducible measured HU; it deliberately does NOT assign
    a diagnostic class (osteoporosis/osteopenia thresholds are ROI-method- and
    cohort-specific, to be clinically validated later). We surface only a soft
    Pickhardt L1 anchor as context. `classification` is kept as a key (None) for
    output-schema stability.
    """
    factor = float(cal.get("factor", 1.0))
    for lvl, r in results.items():
        if r.qc.get("qc_status") == "excluded" or not r.stats:
            continue
        med = r.stats.get("median_HU")
        mean = r.stats.get("mean_HU")
        r.stats["calibration_factor"] = round(factor, 4)
        if isinstance(med, (int, float)):
            r.stats["calibrated_median_HU"] = round(med * factor, 1)
        if isinstance(mean, (int, float)):
            r.stats["calibrated_mean_HU"] = round(mean * factor, 1)
        cal_med = med * factor if isinstance(med, (int, float)) else None
        r.stats["classification"] = None      # no diagnostic class by design
        r.stats["hu_context"] = reference_context(lvl, cal_med)
