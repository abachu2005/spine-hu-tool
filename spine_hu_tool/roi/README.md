# `roi/` — from a vertebra label to a measurement region

This package turns a full-vertebra segmentation into the exact set of voxels we
average HU over. It is the heart of the "math for the ruler" philosophy and the
main driver of reproducibility.

| File | What it does |
|------|--------------|
| `body_isolation.py` | Strip posterior elements → keep the vertebral **body**. |
| `distance.py` | Distance-to-surface transforms used to size ROIs and keep them off the cortex. |
| `modes.py` | The ROI placement modes and the user-facing mode registry. |

## Why isolate the body first (`body_isolation.py`)

Trabecular HU must come from the **cancellous core of the vertebral body** — not
the posterior elements (pedicles, lamina, spinous/transverse processes) and not
the dense cortical rind. A morphological **opening** removes the thin pedicle
connections and disconnects the posterior elements; we keep the most-anterior
large component and recover its volume.

The opening radius is **searched adaptively** (candidates 6→2.5 mm) rather than
fixed: too big annihilates a small vertebra, too small leaves posterior elements
attached. We pick the radius whose body/full ratio best lands in a sane band, and
we **never fall back to the whole vertebra** (that would drag the centroid into
the canal). If nothing isolates cleanly, the level is flagged for exclusion
instead of quietly producing a bad number.

## Why these three ROI modes (`modes.py`)

Only three modes are exposed, each answering a different need:

| Exposed name | Internal | What / why |
|--------------|----------|------------|
| **centroid** *(default)* | `centroid_volume_sphere` | Volume-proportional sphere anchored at the mid-body center. **Placement is anatomy-only (HU-independent)** → maximally reproducible. This is the project's primary metric. |
| **cylinder** | `cylinder_volume` | Same reproducibility-first placement, but a QCT-style axial cylinder (circle in-plane, axis along SI). Familiar VOI shape for bone-density work. |
| **westerhoff** | `lowest_attenuation_sphere` | Volume-proportional sphere placed at the **lowest-attenuation** region, per Westerhoff et al. Kept for methodology comparison / literature alignment. |

### The core reproducibility argument
An HU-seeking placement moves with the data: sample the darkest spot and the
answer depends on noise and where that spot happens to be. The earlier
"largest safe sphere at the distance-transform max" gave an **L1 scan-rescan gap
of ~37 HU**. Anchoring at the body **centroid in the local frame** made
placement depend on anatomy alone and cut the gap to **~7 HU**. That is why the
default is anatomy-anchored, and why placement is separated from the value being
measured.

### Sizing and cortical avoidance
Both anatomy-anchored modes are **volume-proportional** (the ROI is a fixed
fraction of the body's volume, so big and small vertebrae are treated
comparably), then **capped by the in-plane cortical clearance** with a
size-proportional margin so the region stays fully inside the trabecular core and
off the endplates. The reported statistic is the **median HU** (robust to the
occasional cortical/vessel voxel).

### Centering
`_mid_body_center` anchors the ROI at the SI midline of the central band and the
**in-plane incenter** (deepest interior point) of that slice, instead of the raw
centroid. This avoids the posterior basivertebral-notch bias that used to pull
the ROI off-center in sagittal/coronal views.
