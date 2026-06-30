"""Independent HU sanity audit: re-derive trabecular HU straight from the raw
scan + segmentation (no pipeline ROI logic), and cross-check against what the
pipeline reported. Also checks global HU calibration and cortical>>trabecular."""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

from spine_hu_tool.io.series_selector import select_ct_series
from spine_hu_tool.io.dicom_loader import load_series
from spine_hu_tool.segmentation.totalseg_runner import vertebra_labels
from spine_hu_tool.geometry.coords import crop_bbox

FOLDER = "Test Data/100007EC"
SEG = os.path.join(FOLDER, ".spine_hu_cache", "a26feee7433c_seg.nii.gz")

best, _ = select_ct_series(FOLDER)
vol = load_series(best.files)
hu = vol.hu
sp = vol.spacing
seg = np.asarray(nib.load(SEG).dataobj).astype(np.int16)

# pipeline-reported medians (from validate_fullres) for cross-check
PIPE = {"T3":189,"T4":222,"T5":204,"T6":152,"T7":127,"T8":141,
        "T9":125,"T10":128,"T11":128,"T12":149}

print("=== GLOBAL HU CALIBRATION SANITY ===")
print(f"shape={hu.shape} spacing={tuple(round(s,2) for s in sp)}")
print(f"min={hu.min():.0f} max={hu.max():.0f}")
for p in (0.1, 1, 50, 99, 99.9):
    print(f"  p{p}: {np.percentile(hu, p):.0f}")
# air peak should be ~ -1000; soft tissue ~ -50..60
air = hu[hu < -500]
print(f"air voxels median (expect ~ -1000): {np.median(air):.0f}")
soft = hu[(hu > -100) & (hu < 100)]
print(f"soft-tissue band median (expect ~ 0-50): {np.median(soft):.0f}")

print("\n=== PER-LEVEL INDEPENDENT TRABECULAR SAMPLING ===")
print(f"{'lvl':4} {'corebox HU':>22} {'cortex HU':>10} {'pipe med':>8} {'Δ':>6}  note")
present = dict((s, lab) for lab, s in vertebra_labels(seg))
for lvl in PIPE:
    lab = present.get(lvl)
    if lab is None:
        print(f"{lvl:4} (label absent)"); continue
    mask = seg == lab
    sl = crop_bbox(mask, sp, pad_mm=2.0)
    m = mask[sl]; h = hu[sl]
    # spacing-aware erosion: keep voxels >=5 mm inside the mask = trabecular core
    dist = distance_transform_edt(m, sampling=sp)
    core = dist >= 5.0
    rim = m & (dist < 1.5)                      # ~outer shell ~ cortex
    if core.sum() < 20:
        core = dist >= 3.0
    core_hu = h[core]; rim_hu = h[rim]
    med = float(np.median(core_hu))
    p25, p75 = np.percentile(core_hu, [25, 75])
    cortex = float(np.median(rim_hu)) if rim.sum() else float("nan")
    pm = PIPE[lvl]
    delta = med - pm
    note = ""
    if cortex < med:  # cortex should be denser than trabecular core
        note += "CORTEX<CORE? "
    if not (50 < med < 400):
        note += "core out of physiologic band "
    print(f"{lvl:4} {med:7.0f} [{p25:4.0f}-{p75:4.0f}] n={core.sum():5d} "
          f"{cortex:10.0f} {pm:8.0f} {delta:+6.0f}  {note}")
print("\n(core = voxels >=5 mm inside the vertebra mask, a conservative trabecular")
print(" sample independent of the pipeline; pipe med = lowest-attenuation median.)")
