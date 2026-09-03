"""Regenerate validation/REPORT.md from the two original test studies.

Loads each study from DICOM (not from the cached NIfTI) because the scout
body-habitus section needs the volume's real patient-coordinate origin to map
vertebra levels onto scout rows. Segmentations are read from the cached NIfTI
fixtures in work/ so this does not re-run TotalSegmentator.

    python work/regen_validation_report.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

STUDIES = {
    "lumbar (3.75mm STANDARD)": ("Test Data/10000680", "work/lumbar_seg.nii"),
    "thoracic (2.5mm STANDARD)": ("Test Data/100007EC", "work/thoracic_seg.nii"),
}


def main():
    from spine_hu_tool.io.series_selector import select_ct_series
    from spine_hu_tool.io.dicom_loader import load_series
    from spine_hu_tool.segmentation import load_segmentation
    from spine_hu_tool.scout import find_scouts, measure_habitus, level_z_ranges
    from spine_hu_tool.validation import generate_report

    studies, scouts = {}, {}
    for name, (folder, seg_rel) in STUDIES.items():
        folder = os.path.join(ROOT, folder)
        seg_path = os.path.join(ROOT, seg_rel)
        if not (os.path.isdir(folder) and os.path.exists(seg_path)):
            sys.exit(f"missing inputs for {name}: {folder} / {seg_path}")
        print(f"loading {name}...")
        best, cands = select_ct_series(folder)
        volume = load_series(best.files)
        seg = load_segmentation(seg_path)
        studies[name] = (volume, seg)
        found = find_scouts(folder, study_uid=best.study_uid, candidates=cands)
        scouts[name] = measure_habitus(found, level_z_ranges(seg, volume))
        print(f"  scouts: {[s.view for s in found] or 'none'}")

    out = os.path.join(ROOT, "validation", "REPORT.md")
    print("generating report (this measures every level in several ROI modes)...")
    generate_report(studies, out, scouts=scouts)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
