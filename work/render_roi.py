import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
sys.path.insert(0, os.path.dirname(__file__))
import roi_engine as E


def panel(ax, bg, masks, aspect, title):
    ax.imshow(bg, cmap="gray", vmin=-200, vmax=1200, aspect=aspect, origin="lower")
    for m, color, alpha in masks:
        if m.any():
            ax.imshow(np.ma.masked_where(~m, m), cmap=color, alpha=alpha,
                      aspect=aspect, origin="lower", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9); ax.axis("off")


def render(study, level):
    out = E.run(study, level)
    if not out:
        return
    res, H, full, body, info, spacing = out
    sx, sy, sz = spacing
    roi, inner = info["roi"], info["inner"]
    cx, cy, cz = info["center"]
    # axial (fix Z=cz): plane (X,Y) -> show as (Y,X) so anterior up? keep (X,Y).T => (Y,X)
    axial_bg = H[:, :, cz].T
    ax_masks = lambda M: M[:, :, cz].T
    # sagittal (fix X=cx): plane (Y,Z)->T=(Z,Y)
    sag_bg = H[cx, :, :].T
    sg_masks = lambda M: M[cx, :, :].T
    # coronal (fix Y=cy): plane (X,Z)->T=(Z,X)
    cor_bg = H[:, cy, :].T
    co_masks = lambda M: M[:, cy, :].T

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    panel(axes[0], axial_bg,
          [(ax_masks(full), "winter", 0.20), (ax_masks(body), "summer", 0.30),
           (ax_masks(inner), "cool", 0.30), (ax_masks(roi), "autumn", 0.85)],
          aspect=sy / sx, title=f"AXIAL z={cz}  (green=body, blue=inner, red=ROI)")
    panel(axes[1], sag_bg,
          [(sg_masks(full), "winter", 0.20), (sg_masks(body), "summer", 0.30),
           (sg_masks(inner), "cool", 0.30), (sg_masks(roi), "autumn", 0.85)],
          aspect=sz / sy, title="SAGITTAL")
    panel(axes[2], cor_bg,
          [(co_masks(full), "winter", 0.20), (co_masks(body), "summer", 0.30),
           (co_masks(inner), "cool", 0.30), (co_masks(roi), "autumn", 0.85)],
          aspect=sz / sx, title="CORONAL")
    fig.suptitle(f"{study} {level}: mean HU={res['mean_HU']}  r={res['roi_radius_mm']}mm  "
                 f"margin={res['min_margin_mm']:.1f}mm  QC={res['qc_status']}", fontsize=12)
    fig.tight_layout()
    p = f"scratch/roi_{study}_{level}.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    print("rendered", p, "| mean_HU", res["mean_HU"], "| body/full",
          f"{res['body_vox']}/{res['full_vox']}")


if __name__ == "__main__":
    pairs = [("thoracic", "L1"), ("thoracic", "T8"), ("lumbar", "L1")]
    if len(sys.argv) > 2:
        pairs = [(sys.argv[1], sys.argv[2])]
    for s, l in pairs:
        render(s, l)
