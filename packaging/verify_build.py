"""Post-build verification for the FULL OFFLINE installer.

Run this against a freshly built app (in CI or on a per-OS test machine) to
prove the offline local-segmentation runtime is present and usable BEFORE the
build reaches a user. It:

  1. locates the bundled localseg-env + totalseg-weights inside the app,
  2. imports torch + totalsegmentator inside the bundled Python with a sanitized
     environment (the same one the app uses), and
  3. optionally runs a tiny real segmentation to prove offline inference works.

Usage:
    python packaging/verify_build.py --app "packaging/dist/Spine HU Tool.app"
    python packaging/verify_build.py --app "packaging/dist/Spine HU Tool" --run-seg

WARNING: --run-seg performs real segmentation inference and needs enough RAM;
do NOT pass it on a low-memory machine. Without it, the check is import-only and
lightweight (still confirms weights exist for offline use).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spine_hu_tool.segmentation import local_setup as ls  # noqa: E402


def _resolve(app_path: str):
    """Point discovery at the app and return (env_dir, ts_binary, weights_dir)."""
    # macOS .app -> Contents/Resources; onedir -> the dir itself (or _internal).
    candidates = [app_path,
                  os.path.join(app_path, "Contents", "Resources"),
                  os.path.join(app_path, "_internal")]
    for root in candidates:
        env_dir = os.path.join(root, "localseg-env")
        weights = os.path.join(root, "totalseg-weights")
        if os.path.isdir(env_dir):
            os.environ["SPINE_HU_BUNDLED_ENV"] = env_dir
            if os.path.isdir(weights):
                os.environ["SPINE_HU_BUNDLED_WEIGHTS"] = weights
            return env_dir, ls.bundled_ts_binary(), (weights if os.path.isdir(weights) else None)
    return None, None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", required=True, help="path to built .app or onedir dir")
    ap.add_argument("--run-seg", action="store_true",
                    help="also run a tiny real segmentation (needs RAM; offline)")
    args = ap.parse_args(argv)

    env_dir, ts, weights = _resolve(os.path.abspath(args.app))
    if env_dir is None:
        print("FAIL: no bundled localseg-env found in the app.")
        return 1
    print(f"OK: localseg-env at {env_dir}")
    if ts is None or not os.path.exists(ts):
        print("FAIL: bundled TotalSegmentator binary missing.")
        return 1
    print(f"OK: TotalSegmentator at {ts}")
    if weights is None or not any(os.scandir(weights)):
        print("FAIL: bundled totalseg-weights missing or empty (offline run "
              "would try to download).")
        return 1
    print(f"OK: weights at {weights}")

    bindir = "Scripts" if sys.platform.startswith("win") else "bin"
    pyexe = "python" + (".exe" if sys.platform.startswith("win") else "")
    py = os.path.join(env_dir, bindir, pyexe)
    if not os.path.exists(py):
        print(f"FAIL: bundled Python missing at {py}")
        return 1

    # Import torch + totalsegmentator in the bundled Python with the app's
    # sanitized environment (this catches the frozen-app env-leak class of bug).
    env = ls.child_env({"TOTALSEG_HOME_DIR": weights,
                        "TOTALSEG_WEIGHTS_PATH": weights})
    r = subprocess.run([py, "-c", "import torch, totalsegmentator; print(torch.__version__)"],
                       env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("FAIL: bundled Python could not import torch/totalsegmentator:")
        print(r.stderr[-2000:])
        return 1
    print(f"OK: bundled runtime imports (torch {r.stdout.strip()})")

    if args.run_seg:
        print("Running a tiny offline segmentation (this needs RAM)...")
        import numpy as np
        from spine_hu_tool.core import Volume
        from spine_hu_tool.segmentation.totalseg_runner import run_segmentation
        vol = Volume(hu=np.zeros((32, 32, 32), dtype=np.int16), spacing=(3.0, 3.0, 3.0))
        with tempfile.TemporaryDirectory() as tmp:
            seg = run_segmentation(vol, tmp, "verify", fast=True)
        print(f"OK: segmentation returned array shape {seg.shape}")

    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
