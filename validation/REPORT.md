# Spine Vertebral-HU Tool -- Validation Report

Validation uses internal consistency (no external ground truth). Deltas are in Hounsfield Units (HU).

## Levels analyzed per study
- **lumbar (3.75mm STANDARD)**: clean = ['T9', 'T10', 'T11', 'T12', 'L1', 'L2']; excluded (metal) = ['L4', 'L5', 'S1', 'sacrum']
- **thoracic (2.5mm STANDARD)**: clean = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1']; excluded (metal) = ['C6', 'C7']

## Scan-rescan agreement (centroid sphere)
Studies: **lumbar (3.75mm STANDARD)** vs **thoracic (2.5mm STANDARD)**; overlapping clean levels: ['T9', 'T10', 'T11', 'T12', 'L1']

Only levels that pass QC in **both** scans count toward the headline metric; partial/edge or flagged levels are listed but excluded (they are not reportable anyway).

| level | lumbar (3.75mm STANDARD) HU | thoracic (2.5mm STANDARD) HU | |delta| | QC | counted |
|---|---|---|---|---|---|
| T9 | 580.0 | 183.4 | 396.6 | fail/pass | no |
| T10 | 245.1 | 204.7 | 40.4 | fail/pass | no |
| T11 | 208.8 | 194.6 | 14.1 | pass/pass | yes |
| T12 | 209.8 | 191.0 | 18.8 | pass/pass | yes |
| L1 | 178.1 | 185.3 | 7.2 | pass/review | no |

- Mean absolute scan-rescan delta (QC-pass levels only): **16.5 HU** (median 16.5, max 18.8, n=2).
- Baseline (old largest-safe argmax method) L1 delta was ~37 HU; the centroid sphere brings agreement to within kernel/resolution noise.

## ROI-mode agreement
Study **lumbar (3.75mm STANDARD)**, levels ['T9', 'T10', 'T11', 'T12', 'L1', 'L2']

| level | centroid | largest_safe | trabecular_core |
|---|---|---|---|
| T9 | 580.0 | 510.0 | 510.0 |
| T10 | 245.1 | 183.9 | 221.9 |
| T11 | 208.8 | 209.6 | 212.5 |
| T12 | 209.8 | 193.9 | 209.9 |
| L1 | 178.1 | 146.1 | 179.8 |
| L2 | 172.7 | 175.4 | 172.8 |

## Literature anchoring (L1 trabecular)
- Osteoporosis threshold ~110 HU; normal >160 HU (approx; Pickhardt et al.).
- lumbar (3.75mm STANDARD) L1 = 178 HU -> normal range.
- thoracic (2.5mm STANDARD) L1 = 185 HU -> normal range.

## Limitations and intended use
- **Intended use:** a physician-reviewed screening/quantification aid, not a standalone diagnostic device.
- Trabecular HU depends on scanner, reconstruction kernel, kVp, and contrast; compare only within consistent protocols. Report STANDARD-kernel non-contrast values for thresholds.
- Segmentation (TotalSegmentator) is pretrained and used off the shelf; every measurement geometry step is deterministic and physician-auditable.
- Levels with hardware or within the metal streak buffer are excluded automatically and must not be reported.
- Vertebrae touching the scan field edge are flagged 'partial' and should be reviewed (e.g. L1 at the inferior edge of a thoracic scan).
- No external ground-truth (e.g. DXA/QCT phantom) calibration is performed.
