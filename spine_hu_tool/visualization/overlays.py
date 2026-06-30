"""Greyscale tri-planar overlay rendering (axial / sagittal / coronal).

Used for QC PNG export and as the reference look for the review UI: CT in
greyscale with low-saturation translucent overlays for body / inner / ROI.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _panel(ax, bg, mask_layers, aspect, title):
    ax.imshow(bg, cmap="gray", vmin=-200, vmax=1200, aspect=aspect, origin="lower")
    for m, cmap, alpha in mask_layers:
        if m is not None and m.any():
            ax.imshow(np.ma.masked_where(~m, m), cmap=cmap, alpha=alpha,
                      aspect=aspect, origin="lower", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_roi_overlay(hu_crop, result, spacing, out_path: str, level: str = ""):
    """Render axial/sagittal/coronal panels through the ROI center."""
    sx, sy, sz = spacing
    cx, cy, cz = result.center_idx
    full = None  # full mask not retained separately; body suffices
    body = result.body_mask
    inner = result.inner_mask
    roi = result.roi_mask

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    # AXIAL (fix z): plane (x,y) -> show (y,x)
    _panel(axes[0], hu_crop[:, :, cz].T,
           [(body[:, :, cz].T, "summer", 0.30),
            (inner[:, :, cz].T if inner is not None else None, "cool", 0.30),
            (roi[:, :, cz].T, "autumn", 0.85)],
           sy / sx, f"AXIAL z={cz}")
    # SAGITTAL (fix x): plane (y,z) -> (z,y)
    _panel(axes[1], hu_crop[cx, :, :].T,
           [(body[cx, :, :].T, "summer", 0.30),
            (inner[cx, :, :].T if inner is not None else None, "cool", 0.30),
            (roi[cx, :, :].T, "autumn", 0.85)],
           sz / sy, "SAGITTAL")
    # CORONAL (fix y): plane (x,z) -> (z,x)
    _panel(axes[2], hu_crop[:, cy, :].T,
           [(body[:, cy, :].T, "summer", 0.30),
            (inner[:, cy, :].T if inner is not None else None, "cool", 0.30),
            (roi[:, cy, :].T, "autumn", 0.85)],
           sz / sx, "CORONAL")

    s = result.stats
    med = s.get("median_HU", float("nan"))
    cal = s.get("calibrated_median_HU")
    cls = s.get("classification")
    parts = [f"{level or result.level}: median HU={med:.0f}"]
    if cal is not None:
        parts.append(f"calib={cal:.0f}")
    if cls:
        parts.append(f"[{cls}]")
    parts.append(f"r={result.radius_mm:.1f}mm  QC={result.qc.get('qc_status')}")
    fig.suptitle("  ".join(parts), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def render_levels_overlay(hu, seg, present_levels, spacing, out_path: str,
                          title: str = ""):
    """Labeled mid-sagittal overlay of all segmented levels + metal."""
    from .. import config
    sx, sy, sz = spacing
    vmask = seg > 0
    xs = np.where(vmask.any(axis=(1, 2)))[0]
    xc = int(round(xs.mean())) if xs.size else hu.shape[0] // 2
    bg = hu[xc, :, :].T
    seg_sag = seg[xc, :, :].T
    metal_sag = (hu[xc, :, :] > config.METAL_HU_THRESHOLD).T

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(bg, cmap="gray", vmin=-200, vmax=1200, aspect=sz / sy, origin="lower")
    over = np.zeros(seg_sag.shape)
    for i, (lab, short) in enumerate(present_levels):
        over[seg_sag == lab] = i + 1
    cmap = plt.get_cmap("tab20", max(2, len(present_levels) + 1))
    ax.imshow(np.ma.masked_where(over == 0, over), cmap=cmap, alpha=0.45,
              aspect=sz / sy, origin="lower", vmin=0, vmax=len(present_levels))
    if metal_sag.any():
        ax.imshow(np.ma.masked_where(~metal_sag, metal_sag), cmap="autumn",
                  alpha=1.0, aspect=sz / sy, origin="lower")
    for lab, short in present_levels:
        rr, cc = np.where(seg_sag == lab)
        if rr.size:
            ax.text(cc.mean(), rr.mean(), short, color="cyan", fontsize=8,
                    ha="center", va="center", weight="bold")
    ax.set_title(title or "Levels + metal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
