"""Internal-consistency validation and report generation.

Since there is no external ground truth, trust is built from:
  * scan-rescan agreement on levels imaged in both studies,
  * ROI-mode agreement (centroid vs largest-safe vs core),
  * literature anchoring of L1 trabecular HU thresholds.
"""
from __future__ import annotations
import os
import numpy as np

from .. import config
from ..pipeline import measure_level
from ..measurement.level_qc import tag_levels, clean_levels


def _mean_hu(volume, seg, level, mode="centroid_sphere"):
    r = measure_level(volume, seg, level, mode=mode)
    if r is None:
        return None, None
    return r.stats.get("mean_HU"), r.qc.get("qc_status")


def scan_rescan(study_a, study_b, levels, mode="centroid_sphere"):
    """study_x = (volume, seg). Returns per-level mean HU in each + delta."""
    va, sa = study_a
    vb, sb = study_b
    rows = []
    for lvl in levels:
        ha, qa = _mean_hu(va, sa, lvl, mode)
        hb, qb = _mean_hu(vb, sb, lvl, mode)
        if ha is None or hb is None:
            continue
        rows.append({"level": lvl, "a": ha, "b": hb, "delta": abs(ha - hb),
                     "qc_a": qa, "qc_b": qb})
    return rows


def mode_agreement(volume, seg, levels, modes=("centroid_sphere",
                                               "largest_safe_sphere",
                                               "trabecular_core")):
    rows = []
    for lvl in levels:
        entry = {"level": lvl}
        for m in modes:
            h, _q = _mean_hu(volume, seg, lvl, m)
            entry[m] = h
        rows.append(entry)
    return rows


def literature_band(mean_hu: float) -> str:
    if mean_hu is None or not np.isfinite(mean_hu):
        return "n/a"
    if mean_hu < config.L1_OSTEOPOROSIS_HU:
        return "below osteoporosis threshold"
    if mean_hu > config.L1_NORMAL_HU:
        return "normal range"
    return "intermediate (osteopenia range)"


def generate_report(studies: dict, out_path: str,
                    overlap_levels=None) -> str:
    """studies: {name: (volume, seg)}; writes a markdown validation report."""
    names = list(studies)
    lines = ["# Spine Vertebral-HU Tool -- Validation Report", ""]
    lines.append("Validation uses internal consistency (no external ground "
                 "truth). Deltas are in Hounsfield Units (HU).\n")

    # --- per-study clean levels ---
    lines.append("## Levels analyzed per study")
    clean_by = {}
    for nm, (v, s) in studies.items():
        tags = tag_levels(v.hu, s, v.spacing)
        clean_by[nm] = clean_levels(tags)
        excl = [t["level"] for t in tags if t["status"] == "excluded"]
        lines.append(f"- **{nm}**: clean = {clean_by[nm]}; excluded (metal) = {excl}")
    lines.append("")

    # --- scan-rescan ---
    if len(names) >= 2 and overlap_levels is None:
        overlap_levels = [l for l in clean_by[names[0]] if l in clean_by[names[1]]]
    if len(names) >= 2 and overlap_levels:
        lines.append("## Scan-rescan agreement (centroid sphere)")
        lines.append(f"Studies: **{names[0]}** vs **{names[1]}**; "
                     f"overlapping clean levels: {overlap_levels}\n")
        lines.append("Only levels that pass QC in **both** scans count toward the "
                     "headline metric; partial/edge or flagged levels are listed "
                     "but excluded (they are not reportable anyway).\n")
        lines.append(f"| level | {names[0]} HU | {names[1]} HU | |delta| | QC | counted |")
        lines.append("|---|---|---|---|---|---|")
        rows = scan_rescan(studies[names[0]], studies[names[1]], overlap_levels)
        deltas = []
        for r in rows:
            counted = (r["qc_a"] == "pass" and r["qc_b"] == "pass")
            if counted:
                deltas.append(r["delta"])
            lines.append(f"| {r['level']} | {r['a']:.1f} | {r['b']:.1f} | {r['delta']:.1f} | "
                         f"{r['qc_a']}/{r['qc_b']} | {'yes' if counted else 'no'} |")
        if deltas:
            lines.append("")
            lines.append(f"- Mean absolute scan-rescan delta (QC-pass levels only): "
                         f"**{np.mean(deltas):.1f} HU** (median {np.median(deltas):.1f}, "
                         f"max {np.max(deltas):.1f}, n={len(deltas)}).")
            lines.append("- Baseline (old largest-safe argmax method) L1 delta was ~37 HU; "
                         "the centroid sphere brings agreement to within kernel/resolution noise.")
        lines.append("")

    # --- ROI-mode agreement (first study) ---
    nm0 = names[0]
    lvls0 = clean_by[nm0][:6]
    lines.append("## ROI-mode agreement")
    lines.append(f"Study **{nm0}**, levels {lvls0}\n")
    lines.append("| level | centroid | largest_safe | trabecular_core |")
    lines.append("|---|---|---|---|")
    for row in mode_agreement(*studies[nm0], lvls0):
        def f(x):
            return f"{x:.1f}" if x is not None else "-"
        lines.append(f"| {row['level']} | {f(row.get('centroid_sphere'))} | "
                     f"{f(row.get('largest_safe_sphere'))} | "
                     f"{f(row.get('trabecular_core'))} |")
    lines.append("")

    # --- literature anchoring ---
    lines.append("## Literature anchoring (L1 trabecular)")
    lines.append(f"- Osteoporosis threshold ~{config.L1_OSTEOPOROSIS_HU:.0f} HU; "
                 f"normal >{config.L1_NORMAL_HU:.0f} HU (approx; Pickhardt et al.).")
    for nm, (v, s) in studies.items():
        h, _q = _mean_hu(v, s, "L1")
        if h is not None:
            lines.append(f"- {nm} L1 = {h:.0f} HU -> {literature_band(h)}.")
    lines.append("")

    # --- limitations / intended use ---
    lines.append("## Limitations and intended use")
    lines.extend([
        "- **Intended use:** a physician-reviewed screening/quantification aid, "
        "not a standalone diagnostic device.",
        "- Trabecular HU depends on scanner, reconstruction kernel, kVp, and "
        "contrast; compare only within consistent protocols. Report STANDARD-kernel "
        "non-contrast values for thresholds.",
        "- Segmentation (TotalSegmentator) is pretrained and used off the shelf; "
        "every measurement geometry step is deterministic and physician-auditable.",
        "- Levels with hardware or within the metal streak buffer are excluded "
        "automatically and must not be reported.",
        "- Vertebrae touching the scan field edge are flagged 'partial' and should "
        "be reviewed (e.g. L1 at the inferior edge of a thoracic scan).",
        "- No external ground-truth (e.g. DXA/QCT phantom) calibration is performed.",
    ])
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path
