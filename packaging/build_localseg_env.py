"""Build the self-contained local-segmentation runtime bundled in the installer.

The full offline desktop build ships a prebuilt Python + PyTorch +
TotalSegmentator environment *and* the pretrained model weights inside the app
so a user can segment locally with zero setup (no download, works offline).

This script creates that runtime next to (or inside) the PyInstaller output. It
reuses :func:`spine_hu_tool.segmentation.local_setup.provision_env`, so the
runtime is identical to the one the on-demand installer would build -- just
produced ahead of time by CI and copied into the bundle.

Usage (run in CI / on a build machine per OS -- NOT on a low-RAM dev laptop):

    python packaging/build_localseg_env.py --out packaging/localseg-bundle

Produces:
    <out>/localseg-env/       # uv-managed venv with torch + TotalSegmentator
    <out>/totalseg-weights/   # pretrained weights (full-res + fast)

The PyInstaller spec / per-OS packagers copy these two directories next to the
frozen app (see packaging/spine_hu.spec). At runtime the app discovers them via
spine_hu_tool.segmentation.local_setup.bundled_env_dir / bundled_weights_dir.

IMPORTANT: this downloads/installs multi-GB packages and weights and must run on
a machine that can handle it. It never runs segmentation inference itself.
"""
from __future__ import annotations
import argparse
import os
import sys

# Make the package importable when run from a source checkout.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spine_hu_tool.segmentation.local_setup import provision_env  # noqa: E402


def _progress(msg: str) -> None:
    print(f"[build-localseg] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="packaging/localseg-bundle",
                    help="output directory to hold localseg-env + totalseg-weights")
    ap.add_argument("--tasks", nargs="+", default=["total", "total_fast"],
                    help="TotalSegmentator weight tasks to pre-download "
                         "(default: full-res 'total' + fast 'total_fast')")
    ap.add_argument("--best-effort-weights", action="store_true",
                    help="do not fail the build if a weight download fails "
                         "(default: weight download is fatal so no installer "
                         "ships without weights)")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out)
    env_dir = os.path.join(out, "localseg-env")
    weights_dir = os.path.join(out, "totalseg-weights")
    os.makedirs(out, exist_ok=True)

    _progress(f"Building local-seg env at {env_dir}")
    _progress(f"Weights -> {weights_dir} (tasks: {', '.join(args.tasks)})")
    ts = provision_env(env_dir, weights_dir=weights_dir,
                       weight_tasks=tuple(args.tasks), progress=_progress,
                       weights_fatal=not args.best_effort_weights)
    _progress(f"Done. TotalSegmentator: {ts}")

    # Sanity: weights dir must be non-empty when weights are required.
    if not args.best_effort_weights:
        has_files = any(os.scandir(weights_dir)) if os.path.isdir(weights_dir) else False
        if not has_files:
            _progress("ERROR: weights directory is empty after download.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
