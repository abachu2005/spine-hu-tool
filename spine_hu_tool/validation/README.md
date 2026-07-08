# `validation/` — building trust without a ground truth

There is no external gold-standard label for "the true trabecular HU" of these
vertebrae. So instead of claiming accuracy we can't prove, this package measures
the things that *do* make a measurement trustworthy: **consistency**.

| File | What it does |
|------|--------------|
| `consistency.py` | Scan-rescan agreement, ROI-mode agreement, and literature-anchor checks; report generation. |

> Note: this is different from `segmentation/consistency.py`, which validates a
> single *mask*. This package validates the *measurement's* reproducibility
> across scans and methods.

## The three pillars of trust

1. **Scan-rescan agreement.** For levels imaged in more than one study of the
   same patient, the HU should agree. This is the direct evidence for
   reproducibility — and the metric that drove the centroid-ROI redesign (L1 gap
   ~37 HU → ~7 HU). See `validation/REPORT.md` at the repo root.
2. **ROI-mode agreement.** Different reasonable ROI definitions (centroid vs
   largest-safe vs core) should land in a similar range on the same vertebra; a
   large spread means the number is placement-sensitive and untrustworthy.
3. **Literature anchoring.** Values are sanity-checked against published L1
   trabecular thresholds (Pickhardt) so results are in a physically plausible
   regime.

## Why this is the honest way to validate here

Overclaiming diagnostic accuracy without a labeled reference set would be
misleading. Internal consistency + literature anchoring is what we can defend
today, and the harness is built to ingest a radiologist-labeled reference set
later to add external validation.
