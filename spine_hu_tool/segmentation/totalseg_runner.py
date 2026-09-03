"""Run / cache TotalSegmentator and expose vertebra-label helpers.

TotalSegmentator is used off the shelf (pretrained); we never train it. It runs
in a capped-thread CPU subprocess in fast (3 mm) mode by default, which is light
enough for an 8 GB laptop; masks are resampled back to the full-resolution grid
so the deterministic ROI placement is unaffected.
"""
from __future__ import annotations
import os
import sys
import subprocess
import functools
from typing import Optional
import numpy as np

from ..core import Volume
from ..io.nifti_io import save_volume_nifti, load_nifti_array

# Anatomical ordering, superior -> inferior, used for neighbor logic.
VERT_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
              "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
              "T11", "T12",
              "L1", "L2", "L3", "L4", "L5", "S1"]


@functools.lru_cache(maxsize=1)
def _class_map() -> dict:
    """Vertebra/sacrum label-id -> name map.

    Uses the vendored map so the client never needs TotalSegmentator (or torch)
    installed just to interpret a cached mask. If TotalSegmentator is installed
    (optional ``local-seg`` extra), its identical "total" ids are used instead
    so we always stay in sync with the locally-run model.
    """
    try:
        from totalsegmentator.map_to_binary import class_map
        return class_map["total"]
    except Exception:
        from .labels import VERTEBRA_CLASS_MAP
        return dict(VERTEBRA_CLASS_MAP)


@functools.lru_cache(maxsize=1)
def _id_by_name() -> dict:
    return {v: k for k, v in _class_map().items()}


def label_id_for(level: str) -> int:
    """Map a level short name ('L1', 'S1') to its TotalSegmentator label id."""
    m = _id_by_name()
    if level == "sacrum":
        return m["sacrum"]
    return m[f"vertebrae_{level}"]


def vertebra_labels(seg: np.ndarray) -> list[tuple[int, str]]:
    """Present (label_id, short_name) for vertebrae + sacrum, ordered S->I."""
    cm = _class_map()
    out = []
    for lab in np.unique(seg):
        if lab == 0:
            continue
        nm = cm.get(int(lab), "")
        if nm.startswith("vertebrae_") or nm == "sacrum":
            out.append((int(lab), nm.replace("vertebrae_", "")))
    out.sort(key=lambda t: VERT_ORDER.index(t[1]) if t[1] in VERT_ORDER else 99)
    return out


def load_segmentation(path: str) -> np.ndarray:
    arr, _spacing = load_nifti_array(path)
    return arr.astype(np.int16)


def _ts_binary() -> Optional[str]:
    """Path to the TotalSegmentator console script, or None if not installed.

    The packaged desktop client intentionally ships without TotalSegmentator
    (segmentation is offloaded to the cloud), so this can legitimately be None.
    """
    import shutil
    cand = os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")
    if os.path.exists(cand):
        return cand
    on_path = shutil.which("TotalSegmentator")
    if on_path:
        return on_path
    # The app-managed local-seg environment, installed on demand by the user.
    # Require the env's Python to actually START, not merely exist on disk: a
    # stale/orphaned env would otherwise hide the setup button and then crash
    # every run with an opaque trampoline error.
    from .local_setup import managed_ts_binary, env_python_ok
    ts = managed_ts_binary()
    if ts is not None and not env_python_ok():
        return None
    return ts


def local_seg_available() -> bool:
    """True if TotalSegmentator is installed so segmentation can run locally."""
    return _ts_binary() is not None


def run_segmentation(volume: Volume, work_dir: str, name: str,
                     fast: bool = True, force: bool = False,
                     threads: int | None = None) -> np.ndarray:
    """Run (or load cached) TotalSegmentator multilabel segmentation.

    Runs in a separate subprocess (isolates PyTorch memory from the host app),
    on CPU, with capped threads. `fast=True` (3 mm) keeps memory/CPU low enough
    for an 8 GB laptop -- the masks are resampled back to the full-resolution
    grid, so the deterministic ROI placement is unaffected.

    Returns the label array aligned to `volume` ([x, y, z]).
    """
    os.makedirs(work_dir, exist_ok=True)
    seg_path = os.path.join(work_dir, f"{name}_seg.nii.gz")
    if os.path.exists(seg_path) and not force:
        return load_segmentation(seg_path)

    ts = _ts_binary()
    if ts is None:
        raise RuntimeError(
            "Local segmentation is not installed (or its previous install is "
            "broken) on this computer. To fix it: go back to the start screen "
            "and click 'Set up local segmentation...' to (re)install it -- or "
            "untick 'Run segmentation on this computer' to use the cloud "
            "service instead (needs internet).")

    in_path = os.path.join(work_dir, f"{name}.nii.gz")
    if not os.path.exists(in_path) or force:
        save_volume_nifti(volume, in_path)

    if threads is None:
        # On a dedicated server set SPINE_HU_THREADS to use all cores; on a
        # laptop, default to half so the machine stays responsive.
        env_threads = os.environ.get("SPINE_HU_THREADS")
        if env_threads:
            threads = max(1, int(env_threads))
        else:
            threads = max(1, (os.cpu_count() or 4) // 2)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads),
               nnUNet_def_n_proc=str(threads))

    # Device is configurable so the same code runs on a CPU or a GPU server
    # (set SPINE_HU_DEVICE=gpu on a GPU-backed deployment).
    device = os.environ.get("SPINE_HU_DEVICE", "cpu")
    cmd = [ts, "-i", in_path, "-o", seg_path, "--ml",
           "--device", device, "-nr", str(threads), "-ns", "1", "--quiet"]
    if fast:
        cmd.append("--fast")

    # IMPORTANT: log to a file, not a PIPE. nnU-Net spawns worker processes that
    # inherit the stdout/stderr handles; with a PIPE, subprocess.run() blocks on
    # communicate() until those workers close it, which can deadlock after the
    # main process exits. A file handle avoids that entirely.
    log_path = os.path.join(work_dir, f"{name}_seg.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    if proc.returncode != 0 or not os.path.exists(seg_path):
        # tolerate TS appending .nii when the .gz path was not honored
        alt = seg_path[:-3] if seg_path.endswith(".gz") else seg_path + ".nii"
        if os.path.exists(alt):
            return load_segmentation(alt)
        try:
            tail = open(log_path).read()[-800:]
        except OSError:
            tail = ""
        raise RuntimeError(
            "TotalSegmentator failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"log (tail): {tail}")
    return load_segmentation(seg_path)
