# Spine Vertebral-HU Tool

A deterministic, physician-reviewable tool that measures **trabecular Hounsfield
Units (HU)** in vertebral bodies from CT — an opportunistic bone-density signal.

> **Design philosophy: "ML for the eyes, math for the ruler."**
> The only machine-learning component is vertebra localization/segmentation
> (pretrained [TotalSegmentator](https://github.com/wasserth/TotalSegmentator),
> used off the shelf — never trained by us). Every measurement step — body
> isolation, local axes, ROI placement, HU statistics, and QC — is deterministic
> image processing a physician can audit and override.

> **Docs:** for the design rationale and a per-package guide, see
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Every subpackage under
> `spine_hu_tool/` also has its own `README.md` explaining what it does and why.

## What it does

1. **Ingest** a DICOM folder and auto-select the real axial CT series (rejecting
   scouts, reformats, and secondary captures — data quality only).
2. **Segment** each vertebra (TotalSegmentator) and **tag levels** for metal
   hardware / streak artifact, default-excluding instrumented levels and
   flagging their immediate neighbors for review.
3. **Isolate the vertebral body** from the posterior elements (morphological).
4. Build a **per-vertebra local frame** (LR/AP/SI) so placement is tilt-aware.
5. Place a **centroid-anchored sphere** in the central trabecular bone, sized to
   stay clear of cortex (size-proportional margin + adaptive radius).
6. Compute **HU statistics** and **QC flags** (metal proximity, heterogeneity,
   clipping, partial vertebra, cortical-tail hint) → pass / review / fail.
7. Optionally measure **body habitus** — left-right width and anterior-posterior
   depth per level — from the **scout films**, corrected for beam divergence.
8. **Review** in a greyscale, PACS-style desktop app (include / exclude / adjust).
9. **Export** CSV/JSON, tri-planar overlays, masks (NIfTI), a full
   reproducibility record, and an audit trail.

## Body habitus from the scout films

Tick **"Record scout film measurements"** on the landing page (or drop
`--no-scout` on the CLI) to also get, per vertebral level, the patient's
left-right width, anterior-posterior depth, and effective diameter in
millimeters — useful as an outcomes-research covariate.

This has to come from the scouts: a spine protocol reconstructs a tight,
spine-centered field of view, so the body runs off the edge of every axial slice
and its outline exists only in the ~530 mm-wide localizers. The measurement is
deterministic (relative thresholds, largest-run outline, divergent-beam
correction solved between the AP and lateral views) and reproduces to ~1 % across
the two independent studies of the project's one patient with scouts. Studies
whose de-identified export dropped the localizers are detected up front: the
checkbox disables itself and the HU pipeline runs unchanged. See
[`spine_hu_tool/scout/`](spine_hu_tool/scout/README.md).

## Why a centroid sphere

The earlier "largest safe sphere at the distance-transform maximum" landed in
different anatomical spots across scans, giving an **L1 scan-rescan gap of ~37
HU**. Anchoring the sphere at the body centroid in its local frame is
anatomically standardized and reproducible: on the project test data the L1 gap
dropped to **~7 HU**, and fully-imaged overlapping levels agree within
**~14–19 HU** (kernel/resolution noise). See `validation/REPORT.md`.

## Install (pilot — physicians / labmates)

Download a ready-to-run installer (no Python needed). The full offline build
bundles the segmentation runtime + model weights, so it segments **on your
computer with zero setup**; a cloud option is also available.

**Download page:** https://storage.googleapis.com/spine-hu-tool-downloads/index.html

- **macOS**: open the `.dmg`, drag the app to Applications. First launch:
  right-click → **Open** (unsigned-app workaround).
- **Windows**: run the setup wizard. On SmartScreen: **More info → Run anyway**.
- **Linux**: `chmod +x` the AppImage and run it.

To compute segmentation on your machine, check **Run segmentation on this
computer**. Full resolution is the default; on a low-memory machine the app
warns and offers **Fast (3 mm)**. You can analyze one study, several at once, or
all studies in a folder, and reopen past runs from **View past runs**.

Note: all runs — local or cloud — are archived (inputs, outputs, and logs) to
the hosted service for QA, record keeping, and troubleshooting; use
de-identified data only. `SPINE_HU_NO_ARCHIVE=1` disables archival (dev/test).

> Pilot / research use only — not a validated diagnostic device. In the optional
> cloud mode, scans are uploaded to a shared service, so use de-identified data
> only (no PHI); local mode uploads nothing.

## Install (developers)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or, as a package (provides the spine-hu / spine-hu-gui commands):
pip install -e .
# optional: run TotalSegmentator locally instead of the cloud
pip install -e ".[local-seg]"
```

## Usage

### Desktop review app (physician-facing)

```bash
spine-hu-gui            # or: python -m spine_hu_tool.app.viewer
```

Open a DICOM folder → the axial CT is auto-selected → **Analyze** → review levels
in the tri-planar greyscale viewer (green = pass, amber = review, red =
failed or excluded). Failed and automatically excluded levels remain visible and
tunable, but stay out of reported results until explicitly included. Toggle
body/inner/ROI overlays, change window/level, adjust the ROI radius, click to
move the center, **Include result/Exclude result**, then **Export**.

### Command line

```bash
# Full pipeline from DICOM (runs + caches segmentation):
spine-hu measure --dicom "Test Data/100007EC" --all-clean --out results/thoracic

# Quick run from cached NIfTI volume + TotalSegmentator labels:
spine-hu measure --volume work/thoracic.nii.gz --seg work/thoracic_seg.nii \
                 --level L1 --level T12 --out results/quick
```

Useful flags: `--roi-mode {centroid_volume_sphere,cylinder_volume,lowest_attenuation_sphere}`
(default `centroid_volume_sphere`; exposed in the GUI as **centroid / cylinder /
westerhoff**), `--fast` (3 mm segmentation), `--no-overlays`, `--no-masks`,
`--seg-url <URL>` / `--api-key <KEY>` (offload segmentation to a remote server).

### Cloud segmentation (offload the heavy ML step)

TotalSegmentator is the only memory-heavy step (it OOMs ~8 GB laptops). It can
run on a remote service instead, keeping the local app light — only the CT
volume is uploaded; all measurement, QC, and review stay local.

- **Run the server locally** (for testing): `pip install -e ".[server]"` then
  `spine-hu-server` (listens on `$PORT`, default 8080).
- **Deploy to Google Cloud Run** (scale-to-zero, pay only while segmenting):
  see [`deploy/DEPLOY.md`](deploy/DEPLOY.md).
- **Point the app at it** — set once, then use the GUI/CLI normally:

```bash
export SPINE_HU_SEG_URL=https://spine-hu-seg-xxxx.run.app
export SPINE_HU_API_KEY=your-key        # if the service requires auth
```

  Or paste the URL into the landing-screen field in the desktop app. Backend
  selection is transparent: a configured URL → remote, otherwise local. Results
  are cached identically either way, so switching never re-segments a done case.

## Project structure

```
spine_hu_tool/
  io/            DICOM loading, series selection, NIfTI helpers
  preprocessing/ HU rescale math
  segmentation/  TotalSegmentator runner (the only ML) + level helpers
  geometry/      coordinate transforms, morphology, per-vertebra local axes
  roi/           body isolation, distance transform, ROI placement modes
  scout/         body habitus (width/depth per level) from the localizer films
  measurement/   HU stats, ROI QC, per-level metal/streak tagging
  visualization/ greyscale tri-planar overlay rendering
  export/        CSV/JSON/overlay/mask + reproducibility + audit trail
  app/           review-state backend, PySide6 viewer, CLI
  server/        FastAPI segmentation service (offloadable to Cloud Run)
  validation/    internal-consistency report generation
  tests/         synthetic-phantom unit tests + cached integration tests
deploy/          Dockerfile + Cloud Run deploy guide for the server
```

## Testing

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

Unit tests run on synthetic phantoms (no ML). Integration and series-selection
tests use the cached test data when present and skip otherwise.

## Validation & limitations

- Validation is by **internal consistency** (scan-rescan, ROI-mode agreement)
  and **literature anchoring** (L1 trabecular: ~<110 HU osteoporosis, >160 HU
  normal; Pickhardt et al.) — there is no external ground-truth calibration.
- **Intended use:** a physician-reviewed screening/quantification aid, **not** a
  standalone diagnostic device.
- Trabecular HU depends on scanner, kernel, kVp, and contrast — compare only
  within consistent protocols; report STANDARD-kernel non-contrast values.
- Levels that fail QC or contain hardware default to excluded from reporting but
  retain an editable ROI for physician review; only an explicit reviewer action
  can include them. Partial/edge vertebrae are flagged for review.
- Scout body habitus reproduces to ~1 % across the two independent studies of the
  one patient whose export retained localizers; it has not yet been checked
  across patients, because the later de-identified re-exports dropped theirs.

## Out of scope

PHI/de-identification handling; training a custom segmentation model; large-scale
batch infrastructure; formal regulatory submission (intended use documented only).
