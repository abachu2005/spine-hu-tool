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
| T10 | 217.8 | 198.0 | 19.8 | review/pass | no |
| T11 | 192.5 | 186.3 | 6.2 | pass/pass | yes |
| T12 | 195.7 | 188.1 | 7.6 | pass/pass | yes |
| L1 | 175.6 | 178.6 | 3.1 | pass/review | no |

- Mean absolute scan-rescan delta (QC-pass levels only): **6.9 HU** (median 6.9, max 7.6, n=2).
- Baseline (old largest-safe argmax method) L1 delta was ~37 HU; the centroid sphere brings agreement to within kernel/resolution noise.

## Scout body-habitus scan-rescan agreement
Studies: **lumbar (3.75mm STANDARD)** vs **thoracic (2.5mm STANDARD)**; all levels present in both. Values are millimeters.

The two studies were acquired on different days with different table heights and different coverage, from independently acquired scout pairs, so this exercises the outline threshold, the couch rejection and the magnification correction end to end.

| level | LR width A | LR width B | delta | AP depth A | AP depth B | delta | eff. diameter A | eff. diameter B | delta |
|---|---|---|---|---|---|---|---|---|---|
| T9 | 367.4 | 378.1 | +10.7 | 297.7 | 297.0 | -0.7 | 330.7 | 335.1 | +4.4 |
| T10 | 366.9 | 366.2 | -0.7 | 301.4 | 301.3 | -0.1 | 332.5 | 332.2 | -0.3 |
| T11 | 367.0 | 365.7 | -1.3 | 302.8 | 308.5 | +5.7 | 333.4 | 335.9 | +2.5 |
| T12 | 372.4 | 370.8 | -1.6 | 306.9 | 312.7 | +5.8 | 338.1 | 340.5 | +2.4 |
| L1 | 379.9 | 373.8 | -6.1 | 307.5 | 315.2 | +7.7 | 341.8 | 343.3 | +1.5 |

- **LR width**: bias +0.2 mm, mean absolute difference 4.1 mm (max 10.7, n=5).
- **AP depth**: bias +3.7 mm, mean absolute difference 4.0 mm (max 7.7, n=5).
- **eff. diameter**: bias +2.1 mm, mean absolute difference 2.2 mm (max 4.4, n=5).
- Before the divergent-beam correction the LR width carried an ~8 mm systematic offset between the two studies, whose table heights differ by 12 mm; the correction removes it, which is also the empirical check on the beam-direction sign.

## ROI-mode agreement
Study **lumbar (3.75mm STANDARD)**, levels ['T9', 'T10', 'T11', 'T12', 'L1', 'L2']

| level | centroid_volume_sphere | lowest_attenuation_sphere | axial_ellipse_2d |
|---|---|---|---|
| T9 | - | 362.0 | - |
| T10 | 230.2 | 171.1 | 223.3 |
| T11 | 192.5 | 141.9 | 214.4 |
| T12 | 189.2 | 158.8 | 209.1 |
| L1 | 171.7 | 121.1 | 179.6 |
| L2 | 168.4 | 124.6 | 182.3 |

## Literature anchoring (L1 trabecular)
- Osteoporosis threshold ~110 HU; normal >160 HU (approx; Pickhardt et al.).
- lumbar (3.75mm STANDARD) L1 = 176 HU -> normal range.
- thoracic (2.5mm STANDARD) L1 = 179 HU -> normal range.

## Limitations and intended use
- **Intended use:** a physician-reviewed screening/quantification aid, not a standalone diagnostic device.
- Trabecular HU depends on scanner, reconstruction kernel, kVp, and contrast; compare only within consistent protocols. Report STANDARD-kernel non-contrast values for thresholds.
- Segmentation (TotalSegmentator) is pretrained and used off the shelf; every measurement geometry step is deterministic and physician-auditable.
- Levels with hardware or within the metal streak buffer are excluded automatically and must not be reported.
- Vertebrae touching the scan field edge are flagged 'partial' and should be reviewed (e.g. L1 at the inferior edge of a thoracic scan).
- No external ground-truth (e.g. DXA/QCT phantom) calibration is performed.
- Scout body habitus is measured only where the study retained its localizer series; several de-identified re-exports dropped theirs, so the habitus check above covers one patient scanned twice, not a cross-patient cohort.
- The scout outline is a projection boundary, not a reconstructed skin surface: verify it on the exported scout overlay PNGs before reporting, particularly where arms or shoulders enter the field.
