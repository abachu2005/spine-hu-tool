"""Render T3 (and neighbors) with the ROI, body-mask outline, and cortical
(>400 HU) voxels highlighted, so we can SEE whether the sphere touches cortex."""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

LVL = os.environ.get("LVL", "T3")
r = measure_level(vol, seg, LVL, params=ROIParams(), mode="centroid_volume_sphere")
h = hu[r.crop_slices]
body = r.body_mask
roi = r.roi_mask
cx, cy, cz = r.center_idx
print(f"{LVL}: median {r.stats['median_HU']:.0f} HU, radius {r.radius_mm:.1f} mm, "
      f"center {(cx,cy,cz)}, ROI vox {int(roi.sum())}")
print(f"  ROI HU: p5-p95 {r.stats['p05_HU']:.0f}-{r.stats['p95_HU']:.0f}, max {h[roi].max():.0f}")
print(f"  %>300 {100*(h[roi]>300).mean():.0f}%  %>400 {100*(h[roi]>400).mean():.0f}%")


def panel(ax, img2d, roi2d, body2d, title, asp):
    ax.imshow(img2d.T, cmap="gray", vmin=-200, vmax=800, origin="lower", aspect=asp)
    # cortical (>400 HU) voxels in red, faint
    cort = np.ma.masked_where(~(img2d.T > 400), np.ones_like(img2d.T))
    ax.imshow(cort, cmap="autumn", alpha=0.35, origin="lower", aspect=asp)
    ax.contour(body2d.T, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.contour(roi2d.T, levels=[0.5], colors="yellow", linewidths=1.2)
    ax.set_title(title, fontsize=9); ax.axis("off")

fig, axs = plt.subplots(1, 3, figsize=(13, 5))
panel(axs[0], h[:, :, cz], roi[:, :, cz], body[:, :, cz],
      f"{LVL} AXIAL z={cz}", sp[1] / sp[0])
panel(axs[1], h[cx, :, :], roi[cx, :, :], body[cx, :, :],
      f"{LVL} SAGITTAL x={cx}", sp[2] / sp[1])
panel(axs[2], h[:, cy, :], roi[:, cy, :], body[:, cy, :],
      f"{LVL} CORONAL y={cy}", sp[2] / sp[0])
fig.suptitle(f"{LVL}: ROI(yellow) vs body-mask(cyan) vs cortex >400HU(red)", fontsize=11)
fig.tight_layout()
out = f"work/{LVL}_overlay.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
