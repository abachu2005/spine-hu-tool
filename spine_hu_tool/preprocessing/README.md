# `preprocessing/` — HU rescale math

The smallest package in the project, and deliberately so.

| File | What it does |
|------|--------------|
| `hu_conversion.py` | `apply_rescale(raw, slope, intercept)` → `raw * slope + intercept`. |

## Why it exists at all

CT pixels are stored as raw integers; Hounsfield Units come from
`HU = raw * RescaleSlope + RescaleIntercept`. In the normal path,
**SimpleITK already applies this** when reading a series (see `io/`), so this
helper is not on the hot path.

It exists for two reasons:

1. **Explicit/raw-pixel paths** — anywhere we handle pixel data without going
   through SimpleITK, the conversion is named and in one place, not copy-pasted.
2. **Testability** — the HU math can be unit-tested against known values
   independent of any DICOM reader (see `tests/test_hu_conversion.py`).

Keeping the definition of "what an HU is" isolated and tested is worth a
one-line module: it is the ground truth the entire measurement rests on.
