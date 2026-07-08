# `spine_hu_tool` — package map

The Python package behind the Spine Vertebral-HU Tool. For the big-picture
"what and why", read [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). This file
is the **map of the package** and the **order the pieces run in**.

## Top-level modules

| Module | Role |
|--------|------|
| `core.py` | Shared dataclasses: `Volume`, `LocalAxes`, `ROIResult`. |
| `config.py` | All tunable constants + `ROIParams` (thresholds, margins, ROI sizing, calibration references). One place to audit every magic number. |
| `pipeline.py` | Orchestration: `process_case` (DICOM/volume → measured, reviewable case) and `measure_level` (one vertebra). |
| `__main__.py` | `python -m spine_hu_tool` entry. |

## Subpackages (in pipeline order)

1. **`io/`** — load DICOM, pick the real axial CT series, NIfTI helpers.
2. **`preprocessing/`** — HU rescale math (raw → Hounsfield Units).
3. **`segmentation/`** — the *only* ML: TotalSegmentator (cloud or local),
   plus the consistency gate and label helpers.
4. **`geometry/`** — spacing-aware morphology primitives and the per-vertebra
   local LR/AP/SI frame.
5. **`roi/`** — vertebral-body isolation, distance transforms, and the ROI
   placement modes (centroid / cylinder / westerhoff).
6. **`measurement/`** — HU statistics, per-ROI QC, per-level metal/streak
   tagging, and kVp/scanner calibration + soft literature context.
7. **`visualization/`** — greyscale tri-planar overlay rendering.
8. **`export/`** — CSV/JSON, overlays, masks, reproducibility record, audit trail.
9. **`app/`** — the PySide6 review viewer, the CLI, review-state backend, and
   the validation/reproducibility harness.
10. **`server/`** — the FastAPI segmentation service (offloadable to Cloud Run).
11. **`validation/`** — internal-consistency checks (scan-rescan, ROI-mode).
12. **`tests/`** — synthetic-phantom unit tests + cached integration tests.

## The one rule to remember

**ML for the eyes, math for the ruler.** `segmentation/` is the only place a
learned model touches the data. Everything downstream is deterministic and
auditable.
