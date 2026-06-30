"""Validation / reproducibility harness.

Runs the full pipeline (with calibration + ROI-method comparison) and produces
a structured report: per-level measured + calibrated median HU, a soft
literature context note, and the inter-method HU deltas (the
reproducibility/variance angle).

By design the tool reports a *measurement*, not a diagnosis: no osteoporosis
class is assigned here. Clinical thresholds are ROI-method- and cohort-specific
and are to be validated later against a radiologist-labeled reference set (which
this harness is designed to ingest; currently future work).
"""
from __future__ import annotations
import json
import os


def build_report(case: dict) -> dict:
    cal = case.get("calibration") or {}
    mode = case.get("mode")
    levels = {}
    for lvl, r in case["results"].items():
        s = r.stats
        if r.qc.get("qc_status") == "excluded":
            levels[lvl] = {"qc_status": "excluded",
                           "reason": r.qc.get("exclusion_reason")}
            continue
        entry = {
            "qc_status": r.qc.get("qc_status"),
            "median_HU": s.get("median_HU"),
            "calibrated_median_HU": s.get("calibrated_median_HU"),
            "hu_context": s.get("hu_context"),
        }
        if r.comparison:
            entry["method_deltas_vs_primary"] = {
                m: v.get("delta_median_vs_primary")
                for m, v in r.comparison["methods"].items()}
        levels[lvl] = entry

    measured = [v for v in levels.values() if v.get("qc_status") not in
                ("excluded", None)]
    report = {
        "seg_status": case.get("seg_status"),
        "seg_global_reasons": (case.get("seg_check") or {}).get("global_reasons", []),
        "mode": mode,
        "calibration": cal,
        "n_levels": len(levels),
        "n_measured": len(measured),
        "n_excluded": len(levels) - len(measured),
        "levels": levels,
        "notes": [
            "This tool reports a reproducible measured HU, not a diagnosis; no "
            "osteoporosis class is assigned (thresholds to be validated later).",
            "Clinical accuracy requires a radiologist-labeled reference set "
            "(future work).",
        ],
    }
    return report


def print_report(report: dict) -> None:
    print(f"\n=== Validation / reproducibility report ===")
    print(f"segmentation: {report['seg_status']}  |  mode: {report['mode']}  |  "
          f"calibration factor: {(report['calibration'] or {}).get('factor')}")
    print(f"levels: {report['n_measured']} measured, {report['n_excluded']} excluded")
    print(f"\n{'level':6s} {'medHU':>6s} {'calHU':>6s}  deltas(vs primary)   context")
    print("-" * 78)
    for lvl, e in report["levels"].items():
        if e.get("qc_status") == "excluded":
            print(f"{lvl:6s} {'--':>6s} {'EXCLUDED':>6s}")
            continue
        deltas = e.get("method_deltas_vs_primary", {})
        dtxt = " ".join(f"{m.split('_')[0]}:{d:+.0f}" for m, d in deltas.items()
                        if d is not None)
        med = e.get("median_HU")
        cal_med = e.get("calibrated_median_HU")
        print(f"{lvl:6s} {med if med is not None else float('nan'):6.0f} "
              f"{cal_med if cal_med is not None else float('nan'):6.0f}  "
              f"{dtxt:<20s} {e.get('hu_context') or ''}")
    for n in report["notes"]:
        print(f"  note: {n}")


def write_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "validation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
