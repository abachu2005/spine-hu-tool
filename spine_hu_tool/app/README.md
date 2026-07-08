# `app/` — the physician-facing surface

Everything a user actually touches: the desktop review viewer, the CLI, the
review-state backend that both share, and the validation/reproducibility harness.

| File | What it does |
|------|--------------|
| `viewer.py` | The PySide6 greyscale, PACS-style review app. |
| `review_state.py` | Backend state model shared by the GUI and CLI (levels, ROIs, decisions). |
| `cli.py` | `spine-hu measure ...` command-line pipeline. |
| `analysis.py` | Glue that runs `process_case` for the app and packages results. |
| `validate.py` | Reproducibility/validation report harness. |

## Why a custom desktop app (not a notebook or web page)

The users are physicians, and the deliverable is a **review**, not just a number.
A PACS-style greyscale viewer meets them where they already work: tri-planar
views, window/level, and per-level status coloring (**green = pass, amber =
review, red = excluded/metal**). Review is a first-class pipeline stage — the
tool proposes, the physician disposes.

Interaction is built around trust and speed:

- toggle body / inner / ROI overlays and adjust the ROI radius,
- **click-and-drag to move** the ROI center; **Cmd+Z** to undo,
- **Enter** to accept the current level and advance,
- accept/reject are **disabled on excluded levels** (you can't sign off on metal),
- excluded levels still **render** (with the underlying image) so hardware is
  visible, but show no ROI.

The GUI also exposes the **backend choice** (cloud default vs a one-click
"set up local segmentation") and the **series chooser** for parent-folder uploads.

## Why share `review_state.py` between GUI and CLI

The GUI and CLI must produce **identical** measurements — the UI is a viewer over
the same computation, not a second implementation. Both drive `process_case`
through the same state model, so a number reviewed in the app equals a number
produced on the command line.

## Why `validate.py` reports a measurement, not a diagnosis

The harness runs the full pipeline with calibration and the ROI-method
comparison, and reports per-level measured + calibrated median HU with a soft
literature context note and inter-method deltas. It intentionally **does not
assign an osteoporosis class**: clinical thresholds are method-/cohort-specific
and are future work to validate against a radiologist-labeled reference set
(which this harness is designed to ingest).
