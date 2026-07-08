# `segmentation/` — the only machine learning

This is the **one** place a learned model touches the data. Everything else in
the project is deterministic. The package finds and labels vertebrae, decides
*where* to run the model, and guards against the model being wrong.

| File | What it does |
|------|--------------|
| `totalseg_runner.py` | Runs [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (as a subprocess), resolves the binary, and exposes label helpers (`vertebra_labels`, `VERT_ORDER`). |
| `backends.py` | Chooses backend: **cloud by default**, or local. Handles GCS signed-URL upload, async start/poll, retries, API key, and the shared cache. |
| `consistency.py` | The **segmentation consistency gate** — validates the mask before we trust it. |
| `labels.py` | Mapping between TotalSegmentator label ids and vertebra names (C/T/L/S). |
| `local_setup.py` | Self-contained, on-demand install of a local segmentation runtime. |

## Why TotalSegmentator, off the shelf

Finding and separating vertebrae is exactly the fuzzy, perception-style task ML
is good at — and TotalSegmentator is a well-validated, pretrained model. We use
it **as-is and never train it**, so there is no dataset bias we introduced, no
training seed, and no drift. It gives us *the eyes*; the deterministic pipeline
is *the ruler*.

## Why run it as a subprocess

TotalSegmentator pulls in PyTorch and heavy weights. Running it in a subprocess
(a) keeps its memory out of the GUI process, (b) doesn't hold the Python GIL, so
status polling and the UI stay responsive, and (c) lets the same code path drive
either a local install or the cloud service.

## Why a segmentation consistency gate (`consistency.py`)

A model can return a **plausible-looking but wrong** mask — merged levels,
missing levels, out-of-order labels, overlap — especially near metal. With no
ground truth, we sanity-check the geometry the model *should* obey:

- per-label **volume** within an anatomical range (rejects merged/over-grown blobs),
- **SI ordering** of labels (spine goes top-to-bottom),
- **overlap** between non-adjacent labels,
- **contiguity** and spacing sanity.

The whole segmentation is marked `ok` / `suspect` / `invalid`, and individual
levels can be flagged invalid so they are excluded rather than silently
mismeasured.

## Why cloud by default (`backends.py`)

TotalSegmentator is the only step that needs real RAM/GPU; it OOMs ~8 GB
laptops. Offloading it keeps the desktop app lean and installable by anyone.
Backend selection is transparent:

- a configured segmentation URL → **remote** (only the CT volume is uploaded;
  all measurement/QC/review stay local),
- otherwise → **local**.

Robustness details that live here: **GCS signed-URL upload** (bypasses Cloud
Run's 32 MiB request limit), **async start + poll** (long CPU runs survive
dropped idle connections), **retry with exponential backoff** on transient 5xx,
an embedded **shared pilot API key** fallback, and a **Series-UID-keyed cache**
so a finished case never re-segments regardless of backend.

## Why an on-demand local runtime (`local_setup.py`)

A frozen desktop app has nowhere to `pip install`, and we refuse to require a
preinstalled Python. The one-time "set up local segmentation" step therefore
bootstraps [`uv`](https://github.com/astral-sh/uv) (a single static binary),
which **downloads its own standalone CPython** and installs CPU PyTorch +
TotalSegmentator + weights into a user-writable env *outside* the app bundle.
Result: the installer stays ~130 MB, but any labmate can opt into fully-local,
no-upload processing with one click and an internet connection.
