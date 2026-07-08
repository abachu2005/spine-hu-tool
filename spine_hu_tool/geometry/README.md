# `geometry/` — coordinates, morphology, local frames

Spacing-aware geometry primitives shared by body isolation and ROI placement.
Everything here works in **physical millimeters**, never raw voxel counts,
because CT voxels are anisotropic (see [`docs/ARCHITECTURE.md` §4](../../docs/ARCHITECTURE.md)).

| File | What it does |
|------|--------------|
| `coords.py` | Morphology + shape primitives: `ball`, `make_sphere`, `make_cylinder`, `within_mm`, `largest_cc`, `crop_bbox`, distance transforms. |
| `local_axes.py` | The **per-vertebra local LR/AP/SI frame** and central-height band. |

## Why spacing-aware primitives

A "5 mm sphere" or a "3 mm safety margin" is meaningless in voxels when
in-plane pixels are ~0.5 mm and slices are ~3 mm. Every primitive here takes
`spacing` so a millimeter is a millimeter in all three directions. This is why,
e.g., `make_cylinder` builds a shape that is a **circle in the axial plane and a
rectangle in sagittal/coronal**, but is defined in mm on both.

## Why per-vertebra local axes (`local_axes.py`)

The spine is curved (kyphosis, lordosis, scoliosis) and individual vertebrae are
tilted relative to the scan axes. Concepts the measurement depends on —
"the central slice", "stay away from the endplates", "the mid-body" — are only
correct in the vertebra's **own** orientation.

So we derive a local frame per vertebra (principal axes of the body mask, snapped
to the closest global LR/AP/SI directions to stay stable and interpretable) and
compute the **central-height band** in that frame. This makes ROI placement
tilt-aware and keeps the sphere/cylinder off the endplates even on a wedged or
angled level. `crop_bbox` keeps all of this fast by working on a padded crop
around each vertebra instead of the whole volume.
