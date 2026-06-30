"""Measure the ACTUAL clearance from the placed ROI sphere to (a) the body-mask
surface and (b) bright cortical bone, separated by anatomical direction, to see
whether 'cortical contact' is real geometry or an anisotropic-rendering illusion.
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_dilation

from spine_hu_tool.io.series_selector import select_ct_series
from spine_hu_tool.io.dicom_loader import load_series
from spine_hu_tool.config import ROIParams
from spine_hu_tool.pipeline import measure_level

FOLDER = "Test Data/100007EC"
SEG = os.path.expanduser("~/.cache/spine_hu_tool/seg/a26feee7433c_seg.nii.gz")
best, _ = select_ct_series(FOLDER)
vol = load_series(best.files)
seg = np.asarray(nib.load(SEG).dataobj).astype(np.int16)
sp = vol.spacing
hu = vol.hu

for lvl in ("T10", "T3", "T12"):
    r = measure_level(vol, seg, lvl, params=ROIParams(), mode="centroid_volume_sphere")
    if r is None or not r.stats:
        print(f"{lvl}: not measured"); continue
    h = hu[r.crop_slices]
    body = r.body_mask
    roi = r.roi_mask
    cx, cy, cz = r.center_idx
    rad = r.radius_mm

    # half-extent of the body through the center along each axis (mm)
    xs = np.where(body[:, cy, cz])[0]; ys = np.where(body[cx, :, cz])[0]; zs = np.where(body[cx, cy, :])[0]
    def half(idx, c, s):
        return (idx.min() and 0, ) if idx.size == 0 else ((c - idx.min()) * s, (idx.max() - c) * s)
    hx = ((cx - xs.min()) * sp[0], (xs.max() - cx) * sp[0]) if xs.size else (0, 0)
    hy = ((cy - ys.min()) * sp[1], (ys.max() - cy) * sp[1]) if ys.size else (0, 0)
    hz = ((cz - zs.min()) * sp[2], (zs.max() - cz) * sp[2]) if zs.size else (0, 0)

    # gap from sphere edge to body surface, per direction = half-extent - radius
    print(f"\n=== {lvl}: radius {rad:.1f} mm, median {r.stats['median_HU']:.0f} HU, "
          f"vol {r.stats['volume_mm3']:.0f} mm3, center idx {r.center_idx} ===")
    print(f"  body half-extent (mm)  R/L(x): {hx[0]:.1f}/{hx[1]:.1f}   "
          f"A/P(y): {hy[0]:.1f}/{hy[1]:.1f}   S/I(z): {hz[0]:.1f}/{hz[1]:.1f}")
    print(f"  sphere->bodysurf gap   R/L: {hx[0]-rad:+.1f}/{hx[1]-rad:+.1f}   "
          f"A/P: {hy[0]-rad:+.1f}/{hy[1]-rad:+.1f}   S/I: {hz[0]-rad:+.1f}/{hz[1]-rad:+.1f}")

    # does the ROI sit adjacent to the body-mask edge anywhere?
    edge = body & ~binary_dilation(np.pad(body, 1)[1:-1, 1:-1, 1:-1] == body, iterations=0)
    not_body = ~body
    roi_touch_bg = (binary_dilation(roi) & not_body).sum()
    # distance from each ROI voxel to nearest non-body; min = closest approach
    dbg = distance_transform_edt(body, sampling=sp)
    min_clear = dbg[roi].min()
    # cortical HU near the ROI
    cortex = h > 300
    roi_dil = binary_dilation(roi, iterations=2)
    cortex_adj = (roi_dil & cortex & ~roi).sum()
    print(f"  closest ROI-voxel approach to body surface: {min_clear:.1f} mm")
    print(f"  ROI voxels with a non-body neighbor: {roi_touch_bg}  "
          f"(>0 means literal mask contact)")
    print(f"  >300HU voxels within 2 vox of ROI (outside it): {cortex_adj}")
