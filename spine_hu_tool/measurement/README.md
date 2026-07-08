# `measurement/` — the number, the checks, and the context

Given an ROI, produce the HU value; decide whether to trust it; keep metal out;
and make values comparable across scanners without overclaiming a diagnosis.

| File | What it does |
|------|--------------|
| `hu_stats.py` | HU statistics over the ROI mask (mean, **median**, SD, percentiles, volume). |
| `qc.py` | Per-ROI quality control → `pass` / `review` / `fail`. |
| `level_qc.py` | Per-level metal/streak tagging and clean-level selection. |
| `calibration.py` | kVp/scanner HU calibration + soft literature context (and optional classification). |

## The measurement itself is intentionally boring (`hu_stats.py`)

Once the ROI is placed correctly, the measurement is just statistics over those
voxels. We report the **median HU** as the primary value because it is robust to
the occasional stray cortical/vessel voxel; mean/SD/percentiles/volume come along
for QC and transparency. All the intelligence is in *where* the voxels are, not
in a clever estimator.

## Why QC is geometry-first, HU-second (`qc.py`)

Cortical avoidance is guaranteed by **geometry** (the ROI is built to sit inside
the trabecular core). The HU-based checks are a **secondary sanity net**:

- **cortex clipping** — ROI lost more than expected vs its intended shape,
- **metal proximity** — any ROI voxel within the streak buffer of metal → **fail**,
- **heterogeneity** — high SD,
- **cortical-tail hint** — a high p95 suggests a cortical voxel slipped in,
- **size** — too-small ROI or sub-floor safe radius → **fail**.

Result: `pass` (clean), `review` (worth a human look), or `fail` (do not report).

## Why metal is handled at the level, not just the ROI (`level_qc.py`)

Hardware and its **streak/bloom** artifact wreck HU well beyond the metal itself.
So per level we tag:

- metal voxels **inside** the vertebra → **excluded** (never measure through metal),
- within the **streak buffer**, or **immediately adjacent** to an instrumented
  level → **review** (streak corrupts neighbors even when clean-looking).

Excluded levels are still **shown** in review (so the physician sees the
hardware), they just carry no HU.

## Why calibrate but not diagnose (`calibration.py`)

Trabecular HU depends on **tube voltage** (80 vs 120 kVp differ ~23%) and, less,
on **scanner model** (<10%). Following Westerhoff et al. (2025) we apply linear
factors to a 120 kVp / reference-scanner equivalent so values are comparable
across protocols.

For **context only**, we surface where an L1 value falls relative to the widely
cited Pickhardt opportunistic-CT anchors (~<110 HU osteoporosis, >160 HU normal).
A `classify()` helper exists for the Westerhoff per-vertebra thresholds, but the
**main flow deliberately reports a measurement, not a diagnosis** — diagnostic
thresholds are ROI-method- and cohort-specific and must be validated against a
labeled reference set first. Calibration is about comparability; the clinical
call stays with the physician.
