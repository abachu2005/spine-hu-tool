"""Internal HU calibration check near the T3 level: external air, subcutaneous
fat, and paraspinal muscle have known HU. If they read true, a high vertebral
HU is real bone, not a scanner offset."""
import os
import numpy as np
import nibabel as nib
from spine_hu_tool.io.series_selector import select_ct_series
from spine_hu_tool.io.dicom_loader import load_series
from spine_hu_tool.pipeline import measure_level
from spine_hu_tool.config import ROIParams

FOLDER = "Test Data/100007EC"
SEG = os.path.expanduser("~/.cache/spine_hu_tool/seg/a26feee7433c_seg.nii.gz")
best, _ = select_ct_series(FOLDER)
vol = load_series(best.files); hu = vol.hu; sp = vol.spacing
seg = np.asarray(nib.load(SEG).dataobj).astype(np.int16)

# slice at T3 center
r = measure_level(vol, seg, "T3", params=ROIParams(), mode="centroid_volume_sphere")
cz = r.center_idx[2]
sl = hu[:, :, cz]
nx, ny = sl.shape

def box(x0, x1, y0, y1):
    return sl[x0:x1, y0:y1].ravel()

# external air: image corners (well outside the body)
air = np.concatenate([box(0, 20, 0, 20), box(nx-20, nx, 0, 20),
                      box(0, 20, ny-20, ny), box(nx-20, nx, ny-20, ny)])
print(f"T3 slice z={cz}, shape {sl.shape}")
print(f"external-air corners: median {np.median(air):.0f} HU  (true ~ -1000)")

# fat: voxels in [-130,-30] anywhere on the slice (subcutaneous/mediastinal fat)
fat = sl[(sl > -130) & (sl < -30)]
print(f"fat band [-130,-30]:   median {np.median(fat):.0f} HU  (true ~ -90 to -50)")
# muscle/soft tissue: [20,70]
mus = sl[(sl > 20) & (sl < 70)]
print(f"muscle band [20,70]:   median {np.median(mus):.0f} HU  (true ~ 40-50)")
print(f"\nT3 trabecular ROI median: {r.stats['median_HU']:.0f} HU")
# whole-volume air for reference
ext = hu[hu < -200]
print(f"volume air<-200 p50: {np.median(ext):.0f}  (lung-inclusive)")
