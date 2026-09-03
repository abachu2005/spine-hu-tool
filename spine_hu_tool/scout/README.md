# `scout/` — body habitus from the localizer films

This package measures the patient's **body size at each vertebral level** — the
left-right width and the anterior-posterior depth, in millimeters — from the CT
scout (localizer) projections. It is an adjunct to the HU measurement, intended
for outcomes research where body habitus is a covariate.

| File | What it does |
|------|--------------|
| `loader.py` | Find the study's localizer series and load them with the geometry needed to measure in mm (`ScoutImage`). |
| `thickness.py` | Extract the body outline per row, correct for beam divergence, and reduce to per-level width/depth + QC. |

## Why the scout, and not the axial images

A spine protocol reconstructs a **tight, spine-centered field of view** — 280 mm
on the project's lumbar study, 246 mm on the thoracic one. The patient is wider
than that, so the flanks and the belly run off the edge of every axial slice:

```
axial, z = 106.8 mm   body y-extent  -103.3 .. 135.1   <- both ends clipped by the 280 mm FOV
scout  , z = 106.8 mm body y-extent  -145   .. 152     <- the real skin surface
```

The body outline simply does not exist in the reconstructed volume. The scouts
span the full ~530 mm scan field, so they are the only place it can be measured.
This is also why the feature cannot be back-applied to studies whose
de-identified export dropped the localizers.

## The measurement

### Everything is referenced, not absolute

A scout is a **projection radiograph**: pixel values are line integrals, not HU.
Air does not sit at −1000 — on the project's scanners the background is ≈ −450
and the body peaks near +700. So the outline threshold is expressed relative to
two quantities measured from the image itself:

* **background**, the median of the untouched columns at both image edges;
* **body reference**, a high percentile of (value − background), taken **per row**
  and smoothed along z.

The per-row reference is what makes two scans of the same patient comparable. A
thoracic scout that includes the shoulders has a much higher peak attenuation
than a lumbar one; with a single global reference the threshold rises with it and
the same patient measures ~10 mm narrower on the study that includes the
shoulders. Referencing each row to its own anatomy removed that bias (see below).

The body edge is then the **largest contiguous run** above 15 % of the row's
reference.

### The CT couch needs no special model

In a **lateral** projection the beam runs edge-on along the length of the table,
so the couch registers as a distinct, flat, low plateau posterior to the patient
— about 8 % of body attenuation, well under the 15 % body threshold. Taking the
*largest* run additionally rejects the table rails, which spike above the
plateau but are only a few pixels wide. In an **AP** projection the beam crosses
the couch perpendicular, a path of a few millimetres, and it is invisible — which
is exactly what the data shows. So one threshold rule handles both views, and the
exported overlays let a reviewer confirm the outline cleared the couch.

### Magnification

The beam diverges, so an object off the isocenter projects at the wrong scale:

```
apparent = true × SOD / (SOD + d)
```

where `d` is the offset of the body center from the isocenter **along that view's
beam axis**. Each view measures the offset the other one needs — the AP view
measures the left-right center used by the lateral view, and vice versa — so the
pair is solved by a short fixed-point iteration (two passes; the coupling is
weak). Both the corrected and the uncorrected value are reported, along with the
factor applied, so the correction is always auditable.

The correction is small (a few percent) but not cosmetic: it is what removes an
8 mm systematic difference between the project's two studies, whose table heights
differ by 12 mm. That it *cancels* the difference rather than doubling it is also
the empirical check on the sign convention (`SCOUT_BEAM_SIGN`).

### From rows to levels

Each vertebra's segmentation gives its superior-inferior extent in patient
coordinates; the scout and the axial series share the same z frame, so those
bounds map straight onto scout rows. The reported value is the median of the
corrected profile over the **central 60 %** of the level's extent, so rows near
the endplates — where the neighbouring vertebra begins — do not skew it.

## What comes out

Per level, merged into that level's `stats` so they flow through the existing
CSV/JSON/reproducibility writers unchanged:

| Field | Meaning |
|-------|---------|
| `body_width_lr_mm` | Left-right width, from the AP scout |
| `body_depth_ap_mm` | Anterior-posterior depth, from the lateral scout |
| `body_effective_diameter_mm` | √(width × depth) — the AAPM 220 effective diameter |
| `body_*_uncorrected_mm` | The same before magnification correction |
| `scout_{ap,lat}_magnification` | The factor applied |
| `scout_{ap,lat}_center_mm` | Body center offset from the isocenter |
| `scout_band_z_mm` | The z band that was averaged |

Plus `case["scout"]`, carrying the view geometry, the per-image background and
threshold, the parameters, and any warnings.

QC warnings are raised (never silently absorbed) when the outline reaches the
image edge, when the extent falls outside a plausible torso range, when the body
center is far enough off the isocenter that the magnification correction becomes
unreliable, and in two cases worth stating separately:

* **The outline is unstable within the level.** Rows inside one vertebra are
  millimetres apart, so a large peak-to-peak spread across the band means the
  outline left the skin — typically onto an arm lying against the flank — rather
  than the patient changing shape. The reported value is still the band median,
  which is robust to a minority of stray rows, but the level is flagged.
* **The level is in the shoulder girdle.** At C1–T2 the widest thing the AP beam
  crosses is the shoulders and upper arms, not the torso. The number is real and
  reproducible, it simply is not trunk width, so those levels carry an explicit
  anatomical caveat rather than being suppressed.

## Reproducibility

`Test Data/10000680` (lumbar) and `Test Data/100007EC` (thoracic) are the same
patient scanned twice, on different days, with different table heights, different
coverage, and independently acquired scout pairs. At their overlapping levels
(T9–L1) the two measurements agree to:

| Quantity | Bias | Mean abs. difference |
|----------|------|----------------------|
| Left-right width | +0.2 mm | 4.1 mm (1.1 %) |
| Anterior-posterior depth | +3.7 mm | 4.0 mm (1.3 %) |
| Effective diameter | +2.1 mm | 2.2 mm (0.7 %) |

This is the check that `tests/test_scout_integration.py` enforces, because it is
what would break first if a threshold, the couch rejection, or the magnification
correction regressed.

## Failure policy

Body habitus is an adjunct. A missing, odd, or unreadable localizer must never
cost the physician their HU results, so the whole measurement is wrapped at the
pipeline boundary and degrades to `available: false` with a stated reason. In the
GUI the "Record scout film measurements" checkbox disables itself, labelled
*(no scout films in this study)*, when the selected study has no localizers.

## Deliberately out of scope

An in-app scout viewer and manual boundary editing (verification is via the
exported overlay PNGs), water-equivalent thickness, and cohort-scale use — the
last is blocked on re-exports that retain the localizer series.
