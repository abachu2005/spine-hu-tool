# `export/` — results, reproducibility, and audit trail

Writes everything a physician or reviewer needs to use, verify, and defend a
measurement.

| File | What it does |
|------|--------------|
| `writers.py` | CSV/JSON results, ROI-method comparison, `reproducibility.json`, per-level overlays + NIfTI masks, and the append-only `AuditTrail`. |

## What `export_case` produces

- `measurements.csv` / `measurements.json` — the per-level results.
- `overlays/<level>.png` — tri-planar QC images (see `visualization/`).
- `masks/<level>_roi.nii.gz` and `_body.nii.gz` — the exact ROI/body voxels,
  placed back into the **full-volume frame** so they open correctly in any
  DICOM/NIfTI viewer.
- `comparison.{json,csv}` — per-level HU under each ROI method (the
  reproducibility/variance study).
- `reproducibility.json` — see below.

## Why a full reproducibility record

Opportunistic HU is only trustworthy if a number can be **regenerated and
explained**. `reproducibility.json` captures the tool version, the **entire
config** (every threshold/constant), the run params and ROI mode, the
segmentation status and consistency check, the calibration factors, the volume
metadata, and every measurement. Given the same input, the result is
reconstructable bit-for-bit — this is what turns "a number on a screen" into
defensible, auditable evidence.

## Why an append-only audit trail

Physician review is part of the pipeline, so the review actions are part of the
record. `AuditTrail` appends every accept/reject/adjust with reviewer, action,
level, and timestamp. It is append-only by design: the history of decisions is
evidence, not scratch state.
