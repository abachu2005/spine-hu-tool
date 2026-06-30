"""Sweep the cortical-rind standoff and compare each level's ROI median to an
independent deep trabecular core, so we pick a standoff that removes cortical
contamination without collapsing the ROI."""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

from spine_hu_tool.io.series_selector import select_ct_series
from spine_hu_tool.io.dicom_loader import load_series
from spine_hu_tool.segmentation.totalseg_runner import vertebra_labels
from spine_hu_tool.config import ROIParams
from spine_hu_tool.pipeline import measure_level

FOLDER = "Test Data/100007EC"
SEG = os.path.expanduser("~/.cache/spine_hu_tool/seg/a26feee7433c_seg.nii.gz")

import time
t0 = time.time()
best, _ = select_ct_series(FOLDER)
print(f"[{time.time()-t0:.0f}s] series selected", flush=True)
vol = load_series(best.files)
print(f"[{time.time()-t0:.0f}s] volume loaded {vol.hu.shape}", flush=True)
seg = np.asarray(nib.load(SEG).dataobj).astype(np.int16)
sp = vol.spacing
hu = vol.hu

# representative levels: worst contamination (T3/T4), clean mid (T7), lower (T12), edge (L1)
levels = [l for l in ("T3", "T4", "T7", "T12", "L1")
          if any(s == l for _id, s in vertebra_labels(seg))]
print(f"[{time.time()-t0:.0f}s] testing levels {levels}", flush=True)

# independent deep-core median per level (>=5mm inside the FULL body mask)
def deep_core(r):
    body = r.body_mask
    h = hu[r.crop_slices]
    d = distance_transform_edt(body, sampling=sp)
    core = d >= 5.0
    if core.sum() < 20:
        core = d >= 3.0
    return float(np.median(h[core]))

cores = {}
for l in levels:
    ref = measure_level(vol, seg, l, params=ROIParams(cortex_rind_mm=0.0),
                        mode="centroid_volume_sphere")
    cores[l] = deep_core(ref)
    print(f"[{time.time()-t0:.0f}s] core {l}={cores[l]:.0f}", flush=True)

RINDS = [0.0, 2.0, 3.0]
print(f"{'lvl':4} {'core':>5} |" + "".join(f" rind{r:<4}".rjust(13) for r in RINDS))
print("-" * (12 + 13 * len(RINDS)))
for l in levels:
    cells = []
    for rind in RINDS:
        p = ROIParams(cortex_rind_mm=rind)
        res = measure_level(vol, seg, l, params=p, mode="centroid_volume_sphere")
        if res is None or not res.stats:
            cells.append("   --       ")
            continue
        med = res.stats["median_HU"]; rad = res.radius_mm
        f300 = 100.0 * (hu[res.crop_slices][res.roi_mask] > 300).mean()
        d = med - cores[l]
        cells.append(f"{med:4.0f}/r{rad:3.1f}/{f300:2.0f}%".rjust(13))
    print(f"{l:4} {cores[l]:5.0f} |" + "".join(cells))
print("\ncell = ROI_median / radius_mm / %voxels>300HU ; core = deep >=5mm sample")
