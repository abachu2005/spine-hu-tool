"""Force a full-resolution cloud segmentation of the thoracic test scan, using
the corrected (LPS->RAS) NIfTI preprocessing, then run the consistency gate."""
import os
import hashlib
import time

os.environ["SPINE_HU_SEG_URL"] = "https://spine-hu-seg-980966741284.us-central1.run.app"

from spine_hu_tool.io.series_selector import select_ct_series
from spine_hu_tool.io.dicom_loader import load_series
from spine_hu_tool.segmentation.backends import segment
from spine_hu_tool.segmentation.consistency import check_segmentation

FOLDER = "Test Data/100007EC"
best, _ = select_ct_series(FOLDER)
vol = load_series(best.files, metadata={
    "series_uid": best.series_uid, "kvp": best.kvp,
    "manufacturer_model": best.manufacturer_model})
cache = os.path.join(FOLDER, ".spine_hu_cache")
name = hashlib.sha1(best.series_uid.encode()).hexdigest()[:12]

t0 = time.time()
print(f"[fullres] uploading + segmenting (full-res) series={best.description} "
      f"shape={vol.hu.shape} spacing={tuple(round(s,2) for s in vol.spacing)}",
      flush=True)
seg = segment(vol, cache, name, fast=False, force=True,
              progress=lambda m, _f: print(f"[fullres] {m}", flush=True))
print(f"[fullres] done in {time.time()-t0:.0f}s; seg shape={seg.shape}", flush=True)

chk = check_segmentation(seg, vol.spacing)
print(f"[fullres] seg_status={chk['seg_status']}", flush=True)
print(f"[fullres] global_reasons={chk['global_reasons']}", flush=True)
for lvl, v in chk["levels"].items():
    print(f"   {lvl:5} valid={str(v['valid']):5} vol={v['volume_mm3']:8.0f}mm3 "
          f"z=[{v['z_lo_mm']:.0f},{v['z_hi_mm']:.0f}] {v['reasons']}", flush=True)
print("[fullres] DONE", flush=True)
