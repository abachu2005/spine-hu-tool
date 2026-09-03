"""Post-build verification for the FULL OFFLINE installer.

Run this against a freshly built app (in CI or on a per-OS test machine) to
prove the offline local-segmentation runtime is present and usable BEFORE the
build reaches a user. It:

  1. locates the bundled localseg-env + totalseg-weights inside the app,
  2. imports torch + totalsegmentator inside the bundled Python with a sanitized
     environment (the same one the app uses),
  3. SIMULATES A USER MACHINE: copies the bundled runtime to a temp directory
     whose path contains spaces (like "C:\\Program Files\\Spine HU Tool"),
     hides uv's python store, and repeats the health check plus a
     TotalSegmentator --help invocation through the relocated interpreter.
     A runtime that only works in situ -- the v0.2.0 bug, where the bundled
     venv referenced the CI runner's Python -- fails the release here, and
  4. optionally runs a tiny real segmentation to prove offline inference works.

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
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spine_hu_tool.segmentation import local_setup as ls  # noqa: E402


def _resolve(app_path: str):
    """Point discovery at the app and return (env_dir, python, weights_dir)."""
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
            return env_dir, ls.bundled_python(), (weights if os.path.isdir(weights) else None)
    return None, None, None


def _user_machine_env(extra: dict | None = None) -> dict:
    """child_env plus 'this machine has no uv pythons and no dev checkout'."""
    hidden = tempfile.mkdtemp(prefix="no-pythons-")
    env = ls.child_env({"UV_PYTHON_INSTALL_DIR": hidden, **(extra or {})})
    return env


def _check_runtime(py: str, weights: str, label: str) -> bool:
    """Import health check + a TotalSegmentator --help through `py`."""
    env = _user_machine_env({"TOTALSEG_HOME_DIR": weights,
                             "TOTALSEG_WEIGHTS_PATH": weights})
    r = subprocess.run(
        [py, "-c", "import torch, totalsegmentator; print(torch.__version__)"],
        env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"FAIL: {label}: could not import torch/totalsegmentator via {py}:")
        print((r.stderr or r.stdout)[-2000:])
        return False
    print(f"OK: {label}: imports work (torch {r.stdout.strip()})")

    r = subprocess.run(ls.ts_command(py, "--help"),
                       env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"FAIL: {label}: TotalSegmentator --help failed via {py}:")
        print((r.stderr or r.stdout)[-2000:])
        return False
    print(f"OK: {label}: TotalSegmentator entrypoint runs")
    return True


def _check_relocated(env_dir: str, weights: str) -> bool:
    """Copy the runtime to a path WITH SPACES and re-check it there.

    This is the check that would have caught the v0.2.0 bundle: on the build
    machine the original paths still resolve, so an in-situ check passes even
    for a completely non-relocatable runtime.
    """
    with tempfile.TemporaryDirectory(prefix="spine-hu-verify-") as tmp:
        target = os.path.join(tmp, "Relocated App Dir", "localseg-env")
        print(f"Relocating runtime to a spaced path: {target}")
        shutil.copytree(env_dir, target, symlinks=True)
        py = ls.find_runtime_python(target)
        if py is None:
            print("FAIL: no interpreter found in the relocated runtime.")
            return False
        return _check_runtime(py, weights, "relocated (spaced path, no uv pythons)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", required=True, help="path to built .app or onedir dir")
    ap.add_argument("--run-seg", action="store_true",
                    help="also run a tiny real segmentation (needs RAM; offline)")
    args = ap.parse_args(argv)

    env_dir, py, weights = _resolve(os.path.abspath(args.app))
    if env_dir is None:
        print("FAIL: no bundled localseg-env found in the app.")
        return 1
    print(f"OK: localseg-env at {env_dir}")
    if py is None:
        print("FAIL: no Python interpreter found in the bundled runtime.")
        return 1
    print(f"OK: bundled Python at {py}")
    if weights is None or not any(os.scandir(weights)):
        print("FAIL: bundled totalseg-weights missing or empty (offline run "
              "would try to download).")
        return 1
    print(f"OK: weights at {weights}")

    if not _check_runtime(py, weights, "in situ"):
        return 1

    if not _check_relocated(env_dir, weights):
        return 1

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
