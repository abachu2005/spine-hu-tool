"""Validate the new isolation + clipped-sphere ROI on the cached full-res seg.

Geometry only (no HU needed): reports body/full ratio, in-plane safe radius,
final ROI radius + voxel count, and whether the level would be excluded.
"""
import os, sys
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine_hu_tool.geometry.coords import crop_bbox
from spine_hu_tool.roi.body_isolation import isolate_body
from spine_hu_tool.roi.modes import place_roi
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from spine_hu_tool.config import ROIParams

SEG = "Test Data/100007EC/.spine_hu_cache/a26feee7433c_seg.nii.gz"
LEVELS = ["T1", "T9", "T10", "T11", "T12", "L1"]
P = ROIParams()


def main():
    img = nib.load(SEG)
    seg = np.asarray(img.dataobj).astype(np.int16)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    print(f"seg shape={seg.shape} spacing={tuple(round(s,2) for s in spacing)}")
    print(f"{'lvl':4} {'full':>6} {'ratio':>6} {'inplane':>7} {'radius':>6} "
          f"{'roi_vox':>7} {'verdict':>10}  reasons")
    print("-" * 80)
    for lvl in LEVELS:
        try:
            lab = label_id_for(lvl)
        except KeyError:
            continue
        full_f = seg == lab
        if full_f.sum() < 100:
            print(f"{lvl:4} absent")
            continue
        sl = crop_bbox(full_f, spacing)
        full = full_f[sl]
        body, flags = isolate_body(full, spacing)
        info = place_roi(body, spacing, P, "centroid_sphere")
        ratio = body.sum() / max(1, full.sum())
        roi_vox = int(info["roi_mask"].sum())
        vox_mm3 = spacing[0] * spacing[1] * spacing[2]
        vol = roi_vox * vox_mm3

        reasons = []
        if flags.get("partial"): reasons.append("partial")
        if flags.get("too_small") or flags.get("not_isolable"): reasons.append("not-isolable")
        if info["radius_mm"] < P.min_radius_mm: reasons.append("radius<floor")
        if vol < 100.0: reasons.append("vol<min")
        verdict = "EXCLUDED" if reasons else "measured"

        print(f"{lvl:4} {int(full.sum()):6d} {ratio:6.2f} {info['max_safe_radius_mm']:7.2f} "
              f"{info['radius_mm']:6.2f} {roi_vox:7d} {verdict:>10}  {','.join(reasons)}")


if __name__ == "__main__":
    main()
