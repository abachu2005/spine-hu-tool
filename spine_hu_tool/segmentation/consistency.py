"""Deterministic segmentation-validity gate.

TotalSegmentator (like any model) can mislabel vertebral levels, especially on
limited-FOV reconstructions -- producing scattered, overlapping, or speck-sized
labels (e.g. a "T1" that spatially overlaps "T9", or a 37-voxel "L4"). Measuring
such a segmentation yields meaningless numbers, so we screen it first and reject
the invalid levels (or the whole series) before any measurement -- mirroring the
automated consistency checks Westerhoff et al. used to exclude bad series.

All checks are deterministic and parameterized in `config`.
"""
from __future__ import annotations
import numpy as np

from .. import config
from ..geometry.coords import crop_bbox, largest_cc
from .totalseg_runner import vertebra_labels, VERT_ORDER


def _level_geometry(seg: np.ndarray, spacing) -> list[dict]:
    """Per-label geometry (voxels, volume, SI centroid/range, CC fraction)."""
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    out = []
    for lab, short in vertebra_labels(seg):
        mask = seg == lab
        n = int(mask.sum())
        if n == 0:
            continue
        zs = np.where(mask.any(axis=(0, 1)))[0]      # occupied SI (z) indices
        z_center = float(zs.mean()) * spacing[2]
        z_lo, z_hi = float(zs.min()) * spacing[2], float((zs.max() + 1) * spacing[2])
        # a level whose SI extent reaches a z boundary of the scan is truncated
        # by the field of view (expected on partial-spine scans), not mislabeled
        nz = seg.shape[2]
        edge = bool(zs.min() == 0 or zs.max() >= nz - 1)
        # connected-component fraction (on the crop, for speed)
        sl = crop_bbox(mask, spacing, pad_mm=0.0)
        cc = largest_cc(mask[sl])
        cc_frac = float(cc.sum()) / n
        out.append({
            "level": short, "label_id": lab, "voxels": n,
            "volume_mm3": n * voxel_vol, "z_center_mm": z_center,
            "z_lo_mm": z_lo, "z_hi_mm": z_hi, "cc_frac": cc_frac,
            "edge_truncated": edge,
            "rank": VERT_ORDER.index(short) if short in VERT_ORDER else 999,
        })
    return out


def _overlap_frac(a: dict, b: dict) -> float:
    """Fraction of the shorter SI extent that the two levels' z-ranges overlap."""
    lo = max(a["z_lo_mm"], b["z_lo_mm"])
    hi = min(a["z_hi_mm"], b["z_hi_mm"])
    inter = max(0.0, hi - lo)
    shorter = min(a["z_hi_mm"] - a["z_lo_mm"], b["z_hi_mm"] - b["z_lo_mm"])
    return inter / shorter if shorter > 0 else 0.0


def check_segmentation(seg: np.ndarray, spacing) -> dict:
    """Screen a multilabel vertebra segmentation for validity.

    Returns:
      {"seg_status": "ok"|"suspect"|"invalid",
       "levels": {short: {"valid": bool, "reasons": [...], **geom}},
       "global_reasons": [...]}
    """
    geoms = _level_geometry(seg, spacing)
    levels: dict[str, dict] = {}
    global_reasons: list[str] = []

    if not geoms:
        return {"seg_status": "invalid", "levels": {},
                "global_reasons": ["no vertebra labels present"]}

    # --- per-level sanity -----------------------------------------------------
    for g in geoms:
        reasons = []
        if g["volume_mm3"] < config.SEG_MIN_BODY_VOL_MM3:
            reasons.append(f"implausibly small ({g['volume_mm3']:.0f} mm3)")
        if g["volume_mm3"] > config.SEG_MAX_BODY_VOL_MM3:
            reasons.append(f"implausibly large ({g['volume_mm3']:.0f} mm3)")
        if g["cc_frac"] < config.SEG_MIN_CC_FRACTION:
            reasons.append(f"fragmented (largest CC {g['cc_frac']:.2f})")
        levels[g["level"]] = {**g, "valid": not reasons, "reasons": reasons}

    # --- cross-level ordering / overlap (use sane-sized levels only) ----------
    ordered = sorted([g for g in geoms
                      if not levels[g["level"]]["reasons"]],
                     key=lambda g: g["rank"])

    if len(ordered) >= 3:
        zc = np.array([g["z_center_mm"] for g in ordered])
        # anatomical rank increases inferiorly; SI centroid should be monotonic.
        direction = -1.0 if zc[-1] < zc[0] else 1.0
        mono = direction * zc
        inversions = 0
        for i in range(1, len(mono)):
            if mono[i] <= mono[i - 1]:
                inversions += 1
                for g in (ordered[i], ordered[i - 1]):
                    lv = levels[g["level"]]
                    if "SI order inconsistent with level label" not in lv["reasons"]:
                        lv["reasons"].append("SI order inconsistent with level label")
                        lv["valid"] = False
        if inversions:
            global_reasons.append(f"{inversions} SI-ordering inversion(s)")

    # SI overlap between non-adjacent levels (the scrambled-label signature)
    sane = [g for g in geoms if g["volume_mm3"] >= config.SEG_MIN_BODY_VOL_MM3]
    for i in range(len(sane)):
        for j in range(i + 1, len(sane)):
            a, b = sane[i], sane[j]
            if abs(a["rank"] - b["rank"]) <= 1:
                continue                       # adjacent levels may abut
            if _overlap_frac(a, b) > config.SEG_MAX_Z_OVERLAP_FRAC:
                # An overlap between a clean interior level and an FOV-truncated
                # edge level is the truncated/crowded edge label bleeding in --
                # blame only the edge level, not the healthy interior vertebra.
                if a["edge_truncated"] != b["edge_truncated"]:
                    culprits = [g for g in (a, b) if g["edge_truncated"]]
                else:
                    culprits = [a, b]          # both interior or both edge -> both suspect
                for g in culprits:
                    lv = levels[g["level"]]
                    other = b if g is a else a
                    msg = f"SI overlap with non-adjacent {other['level']}"
                    if msg not in lv["reasons"]:
                        lv["reasons"].append(msg)
                        lv["valid"] = False

    # per-step SI spacing between consecutive present (sane) levels
    if len(ordered) >= 2:
        for a, b in zip(ordered, ordered[1:]):
            steps = max(1, abs(b["rank"] - a["rank"]))
            step_mm = abs(b["z_center_mm"] - a["z_center_mm"]) / steps
            if step_mm < config.SEG_MIN_STEP_MM or step_mm > config.SEG_MAX_STEP_MM:
                global_reasons.append(
                    f"{a['level']}-{b['level']} spacing {step_mm:.0f} mm/level implausible")

    # label contiguity (informational: gaps can be legit FOV cropping)
    ranks = sorted(g["rank"] for g in geoms if g["rank"] < 999)
    if ranks and (ranks[-1] - ranks[0] + 1) - len(ranks) >= 3:
        global_reasons.append("non-contiguous level labels (>=3 missing in span)")

    # --- aggregate ------------------------------------------------------------
    # FOV-truncated levels (touching a z boundary) are an acquisition property,
    # not a labeling failure, so they don't count toward "is this seg broken".
    interior = [v for v in levels.values() if not v.get("edge_truncated")]
    n_interior = len(interior)
    n_invalid_interior = sum(1 for v in interior if not v["valid"])
    frac_invalid = n_invalid_interior / max(1, n_interior)
    ordering_broken = any("SI-ordering" in r for r in global_reasons) or \
        any("SI order inconsistent" in rr
            for v in levels.values() for rr in v["reasons"])
    n_invalid = sum(1 for v in levels.values() if not v["valid"])

    if ordering_broken or frac_invalid > config.SEG_INVALID_FRACTION:
        # genuinely scrambled: broken ordering, or many interior levels bad
        seg_status = "invalid"
    elif n_invalid > 0 or global_reasons:
        # a usable core with truncated/edge levels or minor anomalies
        seg_status = "suspect"
    else:
        seg_status = "ok"

    return {"seg_status": seg_status, "levels": levels,
            "global_reasons": global_reasons}
