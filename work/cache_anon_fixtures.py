"""Cache the new Anon studies' selected volume + (already-computed) segmentation
into work/ as small NIfTI fixtures, so the offline integration tests can run
deterministically without DICOM or the cloud.

Anon1 -> anon_longspine    (long C7->sacrum scan; FOV-clipped ends)
Anon2 -> anon_instrumented (spinal hardware at L3/L5/S1)
"""
import os
import shutil
import hashlib

from spine_hu_tool.io.series_selector import select_ct_series
from spine_hu_tool.io.dicom_loader import load_series
from spine_hu_tool.io.nifti_io import save_volume_nifti
from spine_hu_tool.app.analysis import _seg_cache_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
CACHE = _seg_cache_dir()

MAP = {
    "Anon1": "anon_longspine",
    "Anon2": "anon_instrumented",
}

for folder, stem in MAP.items():
    src = os.path.join(ROOT, "Test Data", folder)
    best, _ = select_ct_series(src)
    if best is None:
        print(f"!! no series for {folder}")
        continue
    vol = load_series(best.files, metadata={
        "series_uid": best.series_uid, "kvp": best.kvp,
        "manufacturer_model": best.manufacturer_model,
        "kernel": best.kernel, "slice_thickness": best.slice_thickness})
    name = hashlib.sha1(best.series_uid.encode()).hexdigest()[:12]
    seg_cached = os.path.join(CACHE, f"{name}_seg.nii.gz")
    if not os.path.exists(seg_cached):
        print(f"!! no cached seg for {folder} ({seg_cached}); run the pipeline first")
        continue
    vol_out = os.path.join(WORK, f"{stem}.nii.gz")
    seg_out = os.path.join(WORK, f"{stem}_seg.nii.gz")
    save_volume_nifti(vol, vol_out)
    shutil.copy2(seg_cached, seg_out)
    print(f"{folder} -> {stem}: vol shape={vol.hu.shape} spacing="
          f"{tuple(round(s,3) for s in vol.spacing)} "
          f"kvp={best.kvp} model='{best.manufacturer_model}'")
    print(f"   wrote {vol_out} ({os.path.getsize(vol_out)//1024} KB)")
    print(f"   wrote {seg_out} ({os.path.getsize(seg_out)//1024} KB)")
print("DONE")
