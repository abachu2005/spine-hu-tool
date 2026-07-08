# `visualization/` — greyscale overlays

Renders the CT with translucent overlays so a human can *see* exactly what was
measured. Used both for exported QC PNGs and as the reference look for the
review UI.

| File | What it does |
|------|--------------|
| `overlays.py` | `render_roi_overlay` (tri-planar axial/sagittal/coronal through the ROI center) and `render_levels_overlay` (labeled mid-sagittal of all levels + metal). |

## Why greyscale + low-saturation overlays

Radiologists read CT in greyscale with a fixed window/level; a garish colormap
would fight their trained eye. So the background is greyscale (`vmin=-200`,
`vmax=1200`) and the body/inner/ROI overlays are **low-saturation and
translucent** — visible enough to verify placement, faint enough not to obscure
the anatomy. Metal is drawn opaque so it can't be missed.

## Why tri-planar, through the ROI center

Reproducibility and safety are 3-D properties: an ROI can look centered axially
but clip an endplate in sagittal. Rendering all three orthogonal planes **through
the ROI center** lets a reviewer confirm at a glance that the region is inside
the trabecular core and clear of cortex/endplates in every direction. Panels use
aspect ratios derived from the real voxel spacing so shapes aren't distorted.

## Why a separate "levels + metal" view

`render_levels_overlay` gives the whole-spine picture: every segmented level
labeled on a mid-sagittal slice, with metal highlighted. This is the fast way to
confirm the segmentation is sane and to see which levels are excluded by hardware
before diving into per-level ROIs.
