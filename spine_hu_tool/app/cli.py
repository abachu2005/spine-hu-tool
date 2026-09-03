"""Command-line interface: spine-hu measure ...

Examples:
  spine-hu measure --dicom "Test Data/100007EC" --all-clean --out results/thoracic
  spine-hu measure --volume work/thoracic.nii.gz --seg work/thoracic_seg.nii \\
                   --level L1 --level T12 --out results/quick
"""
from __future__ import annotations
import argparse
import sys

from ..config import ROIParams
from ..io.dicom_loader import load_volume_from_nifti
from ..pipeline import process_case
from ..export.writers import export_case
from ..roi.modes import EXPOSED_MODES, DEFAULT_MODE


def _seg_status_reasons(case) -> list[str]:
    """Human-readable reasons a segmentation was flagged suspect/invalid.

    Prefer the global reasons (ordering/spacing/contiguity), but fall back to a
    compact summary of the per-level failures. Without this, a segmentation that
    is 'suspect' purely because of a few bad *levels* (e.g. a speck-sized T8 or
    an over-segmented sacrum) prints an empty reason, which reads like a bug.
    """
    check = case.get("seg_check") or {}
    reasons = list(check.get("global_reasons") or [])
    if reasons:
        return reasons
    per_level = []
    for lvl, v in (check.get("levels") or {}).items():
        if not v.get("valid", True) and v.get("reasons"):
            per_level.append(f"{lvl}: {'; '.join(v['reasons'])}")
    return per_level


def _print_summary(case):
    seg_status = case.get("seg_status")
    if seg_status and seg_status != "ok":
        reasons = _seg_status_reasons(case)
        detail = "; ".join(reasons) if reasons else "per-level anomalies (see level table)"
        print(f"\n!! SEGMENTATION {seg_status.upper()}: " + detail)
        if seg_status == "invalid":
            print("   Affected levels were excluded; consider re-running/re-acquiring.")
    cal = case.get("calibration") or {}
    if cal:
        print(f"\nCalibration: factor={cal.get('factor')} "
              f"(kVp={cal.get('kvp')}, {', '.join(cal.get('notes', []))})")
    scout = case.get("scout") or {}
    has_habitus = bool(scout.get("levels"))
    if scout and not scout.get("available"):
        for w in scout.get("warnings", []):
            print(f"\nScout: {w}")
    hab_head = f" {'LRmm':>6s} {'APmm':>6s}" if has_habitus else ""
    print(f"\n{'level':6s} {'medHU':>6s} {'calHU':>6s} {'radius':>6s}"
          f"{hab_head} {'QC':>9s}  context")
    print("-" * (56 + len(hab_head)))
    for lvl, r in case["results"].items():
        s = r.stats
        hab = ""
        if has_habitus:
            w = s.get("body_width_lr_mm")
            d = s.get("body_depth_ap_mm")
            hab = (f" {w:6.1f}" if w is not None else f" {'-':>6s}") + \
                  (f" {d:6.1f}" if d is not None else f" {'-':>6s}")
        print(f"{lvl:6s} {s.get('median_HU', float('nan')) or float('nan'):6.1f} "
              f"{s.get('calibrated_median_HU', float('nan')) or float('nan'):6.1f} "
              f"{r.radius_mm:6.1f}{hab} {r.qc.get('qc_status'):>9s}  "
              f"{s.get('hu_context') or ''}")


def cmd_measure(args):
    params = ROIParams()
    levels = args.level or None
    if args.dicom:
        from .analysis import analyze_dataset
        state = analyze_dataset(args.dicom, mode=args.roi_mode,
                                only_clean=(levels is None),
                                fast=(True if args.fast else None),
                                reviewer=args.reviewer, params=params,
                                seg_url=args.seg_url, api_key=args.api_key,
                                compute_comparison=args.compare,
                                apply_calibration=not args.no_calibration,
                                scout=not args.no_scout,
                                progress=lambda m, f: print(f"[{f*100:3.0f}%] {m}"))
        if levels:
            # restrict to requested levels (already segmented & cached)
            restricted = {}
            for l in levels:
                r = state.recompute_level(l)
                if r is not None:
                    restricted[l] = r
            state.case["results"] = restricted
        case, volume = state.case, state.volume
    else:
        if not (args.volume and args.seg):
            print("error: provide --dicom, or both --volume and --seg", file=sys.stderr)
            return 2
        from ..segmentation.totalseg_runner import load_segmentation
        volume = load_volume_from_nifti(args.volume)
        seg = load_segmentation(args.seg)
        case = process_case(volume, seg, levels=levels, params=params,
                            mode=args.roi_mode, only_clean=(levels is None),
                            compute_comparison=args.compare,
                            apply_calibration=not args.no_calibration)

    _print_summary(case)
    written = export_case(case, volume, args.out,
                          overlays=not args.no_overlays, masks=not args.no_masks)
    print(f"\nExported to {args.out}:")
    for k, v in written.items():
        print(f"  {k}: {v}")
    return 0


def _load_case(args, params, compute_comparison, apply_calibration):
    """Resolve --dicom or --volume/--seg into (case, volume)."""
    levels = args.level or None
    if args.dicom:
        from .analysis import analyze_dataset
        state = analyze_dataset(args.dicom, mode=args.roi_mode,
                                only_clean=(levels is None),
                                fast=(True if getattr(args, "fast", False) else None),
                                reviewer=getattr(args, "reviewer", "cli"), params=params,
                                seg_url=getattr(args, "seg_url", None),
                                api_key=getattr(args, "api_key", None),
                                compute_comparison=compute_comparison,
                                apply_calibration=apply_calibration,
                                scout=not getattr(args, "no_scout", False),
                                progress=lambda m, f: print(f"[{f*100:3.0f}%] {m}"))
        if levels:
            restricted = {}
            for l in levels:
                r = state.recompute_level(l)
                if r is not None:
                    restricted[l] = r
            state.case["results"] = restricted
        return state.case, state.volume
    if not (args.volume and args.seg):
        raise SystemExit("error: provide --dicom, or both --volume and --seg")
    from ..segmentation.totalseg_runner import load_segmentation
    volume = load_volume_from_nifti(args.volume)
    seg = load_segmentation(args.seg)
    case = process_case(volume, seg, levels=levels, params=params,
                        mode=args.roi_mode, only_clean=(levels is None),
                        compute_comparison=compute_comparison,
                        apply_calibration=apply_calibration)
    return case, volume


def cmd_validate(args):
    from .validate import build_report, print_report, write_report
    params = ROIParams()
    # validation always compares ROI methods and applies calibration
    case, volume = _load_case(args, params, compute_comparison=True,
                              apply_calibration=not args.no_calibration)
    _print_summary(case)
    report = build_report(case)
    print_report(report)
    written = export_case(case, volume, args.out,
                          overlays=not args.no_overlays, masks=not args.no_masks)
    report_path = write_report(report, args.out)
    written["validation_report"] = report_path
    print(f"\nExported to {args.out}:")
    for k, v in written.items():
        print(f"  {k}: {v}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="spine-hu",
                                description="Trabecular vertebral-HU measurement (physician-reviewed).")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="measure HU per vertebral level")
    src = m.add_argument_group("input (choose one)")
    src.add_argument("--dicom", help="DICOM folder (runs segmentation)")
    src.add_argument("--volume", help="NIfTI HU volume (with --seg)")
    src.add_argument("--seg", help="NIfTI TotalSegmentator labels (with --volume)")
    m.add_argument("--level", action="append", help="level e.g. L1 (repeatable); default = all clean")
    m.add_argument("--all-clean", action="store_true", help="measure all clean levels (default)")
    m.add_argument("--roi-mode", default=DEFAULT_MODE,
                   choices=list(EXPOSED_MODES.values()))
    m.add_argument("--compare", action="store_true",
                   help="also measure every ROI method per level (reproducibility study)")
    m.add_argument("--no-calibration", action="store_true",
                   help="skip kVp/scanner HU calibration (report raw HU)")
    m.add_argument("--no-scout", action="store_true",
                   help="skip the scout-film body-habitus measurement "
                        "(width/depth per level)")
    m.add_argument("--out", required=True, help="output folder")
    m.add_argument("--fast", action="store_true",
                   help="force fast (3mm) segmentation (default: full-res when using a "
                        "cloud --seg-url, fast when segmenting locally)")
    m.add_argument("--seg-url", dest="seg_url", default=None,
                   help="remote segmentation server URL (default: cloud; "
                        "env SPINE_HU_SEG_URL or deploy credentials, else baked-in cloud URL)")
    m.add_argument("--api-key", dest="api_key", default=None,
                   help="API key for the segmentation server (default: env SPINE_HU_API_KEY)")
    m.add_argument("--reviewer", default="cli")
    m.add_argument("--no-overlays", action="store_true")
    m.add_argument("--no-masks", action="store_true")
    m.set_defaults(func=cmd_measure)

    val = sub.add_parser("validate",
                         help="run the validation harness (ROI-method comparison + reproducibility report)")
    vsrc = val.add_argument_group("input (choose one)")
    vsrc.add_argument("--dicom", help="DICOM folder (runs segmentation)")
    vsrc.add_argument("--volume", help="NIfTI HU volume (with --seg)")
    vsrc.add_argument("--seg", help="NIfTI TotalSegmentator labels (with --volume)")
    val.add_argument("--level", action="append", help="level e.g. L1 (repeatable)")
    val.add_argument("--roi-mode", default=DEFAULT_MODE,
                     choices=list(EXPOSED_MODES.values()))
    val.add_argument("--no-calibration", action="store_true")
    val.add_argument("--no-scout", action="store_true")
    val.add_argument("--out", required=True, help="output folder")
    val.add_argument("--fast", action="store_true")
    val.add_argument("--seg-url", dest="seg_url", default=None)
    val.add_argument("--api-key", dest="api_key", default=None)
    val.add_argument("--reviewer", default="cli")
    val.add_argument("--no-overlays", action="store_true")
    val.add_argument("--no-masks", action="store_true")
    val.set_defaults(func=cmd_validate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
