# `io/` — data in

Turns a folder of DICOM files into a clean, correctly-oriented `Volume`, and
handles NIfTI I/O for cached data and segmentation exchange.

| File | What it does |
|------|--------------|
| `series_selector.py` | Scans a folder, groups files by `SeriesInstanceUID`, and **ranks** series so the real axial CT is auto-selected. |
| `dicom_loader.py` | Loads a chosen series into a `Volume` (HU, spacing, origin, LPS direction) via SimpleITK. |
| `nifti_io.py` | Read/write NIfTI (cached volumes, masks, TotalSegmentator exchange). |

## Why series selection is its own step

A single DICOM folder usually contains more than the scan you want: localizer
scouts, reformatted/secondary-capture views, and the *same* acquisition
reconstructed with several kernels (STANDARD, BONE, …). Measuring HU off the
wrong one silently corrupts everything downstream. So `series_selector`:

- **Rejects** non-usable series (not CT, localizer, reformatted, secondary
  capture, too few slices, missing rescale, non-uniform slice spacing).
- **Scores** the rest, preferring a **STANDARD kernel** (correct HU), more
  slices, and thinner slices.
- **Groups by patient → study** so a physician who points the tool at a parent
  folder full of patients still gets a sane chooser, and **collapses redundant
  kernel reconstructions** to one row per real study (`list_studies`) so the
  choice is "Thoracic vs Lumbar", not "STANDARD vs BONE of the same scan".

> Scope: this is **data-quality only** — no PHI/de-identification handling. Use
> de-identified data.

## Why SimpleITK for loading

SimpleITK reads a series robustly and **applies `RescaleSlope`/`RescaleIntercept`
for us**, so values are already HU. We then transpose from its `[z, y, x]` order
into the project's `[x, y, z]` convention and keep the **LPS direction cosines**
so we can write a correctly-oriented NIfTI for TotalSegmentator (a wrong affine
flips left/right and mislabels the spine).
