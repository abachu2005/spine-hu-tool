"""On-demand, self-contained setup of the local segmentation runtime.

The packaged desktop app ships lean (no PyTorch / TotalSegmentator) and segments
on the cloud by default. A user who wants to run segmentation on their own
machine can install the heavy ML stack once into a *managed* environment that
lives in a user-writable data directory (outside the read-only app bundle). The
app then runs that environment's ``TotalSegmentator`` console script as a
subprocess.

To make this work on ANY machine -- with no preinstalled Python -- setup
bootstraps `uv` (a single self-contained binary from Astral), which downloads
its own standalone CPython and creates the environment. So the only requirement
is a one-time internet connection; nothing needs to be installed beforehand.

This module owns: where the environment lives, fetching `uv`, provisioning the
Python + ML stack with streamed progress, and reporting readiness.
"""
from __future__ import annotations
import os
import sys
import platform
import shutil
import subprocess
import tarfile
import zipfile
import tempfile
import urllib.request
from typing import Callable, Optional

ProgressCb = Optional[Callable[[str], None]]

_IS_WIN = sys.platform.startswith("win")

# Pinned Python for the managed env: a version with known torch / TotalSegmentator
# wheels (uv downloads a standalone build of it -- no system Python needed).
_MANAGED_PY = "3.11"


def _user_data_root() -> str:
    """OS-appropriate per-user data dir for app-managed files."""
    if _IS_WIN:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = (os.environ.get("XDG_DATA_HOME")
                or os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(base, "SpineHUTool")


def managed_env_dir() -> str:
    """Directory of the app-managed local-segmentation environment."""
    return os.environ.get("SPINE_HU_LOCALSEG_ENV") or os.path.join(
        _user_data_root(), "localseg-env")


def _venv_bin_dir(env_dir: str) -> str:
    return os.path.join(env_dir, "Scripts" if _IS_WIN else "bin")


def _exe(name: str) -> str:
    return name + ".exe" if _IS_WIN else name


def managed_python() -> Optional[str]:
    """Path to the managed env's Python, or None if the env doesn't exist."""
    cand = os.path.join(_venv_bin_dir(managed_env_dir()), _exe("python"))
    return cand if os.path.exists(cand) else None


def managed_ts_binary() -> Optional[str]:
    """Path to the managed env's TotalSegmentator script, or None."""
    cand = os.path.join(_venv_bin_dir(managed_env_dir()), _exe("TotalSegmentator"))
    return cand if os.path.exists(cand) else None


def is_ready() -> bool:
    """True if the managed local-segmentation runtime is fully installed."""
    py = managed_python()
    if py is None or managed_ts_binary() is None:
        return False
    try:
        r = subprocess.run([py, "-c", "import torch, totalsegmentator"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=180)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --- uv bootstrap (provides a standalone Python; no system Python needed) -----

def _uv_asset() -> str:
    """GitHub release asset name of the `uv` binary for this platform."""
    mach = platform.machine().lower()
    arm = mach in ("arm64", "aarch64")
    if sys.platform == "darwin":
        return f"uv-{'aarch64' if arm else 'x86_64'}-apple-darwin.tar.gz"
    if _IS_WIN:
        return f"uv-{'aarch64' if arm else 'x86_64'}-pc-windows-msvc.zip"
    # linux
    return f"uv-{'aarch64' if arm else 'x86_64'}-unknown-linux-gnu.tar.gz"


def _managed_uv_path() -> str:
    return os.path.join(_user_data_root(), "bin", _exe("uv"))


def _ensure_uv(progress: ProgressCb) -> str:
    """Return a path to a usable `uv`, downloading it if necessary."""
    existing = _managed_uv_path()
    if os.path.exists(existing):
        return existing
    on_path = shutil.which("uv")
    if on_path:
        return on_path

    asset = _uv_asset()
    url = f"https://github.com/astral-sh/uv/releases/latest/download/{asset}"
    if progress:
        progress("Downloading setup tool (uv)...")
    dest_dir = os.path.dirname(existing)
    os.makedirs(dest_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, asset)
        urllib.request.urlretrieve(url, archive)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(tmp)
        # locate the extracted uv binary (release archives nest it in a folder)
        found = None
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                if f == _exe("uv"):
                    found = os.path.join(root, f)
                    break
            if found:
                break
        if found is None:
            raise RuntimeError("Could not find the uv binary in the download.")
        shutil.copy2(found, existing)
    if not _IS_WIN:
        os.chmod(existing, 0o755)
    return existing


def _run(cmd, progress: ProgressCb, label: str, env=None) -> None:
    """Run a subprocess, streaming its output line-by-line to `progress`."""
    if progress:
        progress(label)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    last = ""
    for line in proc.stdout:                       # stream so the UI stays live
        last = line.strip()
        if progress and last:
            progress(last[-160:])
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{last}")


def setup_local_seg(progress: ProgressCb = None) -> str:
    """Provision the managed env and install the local segmentation stack.

    Self-contained: bootstraps `uv`, which downloads a standalone Python, then
    installs a CPU PyTorch + TotalSegmentator and pre-downloads the model
    weights. Returns the managed TotalSegmentator binary path. Safe to re-run.
    """
    if is_ready():
        if progress:
            progress("Local segmentation already installed.")
        return managed_ts_binary()  # type: ignore[return-value]

    env_dir = managed_env_dir()
    os.makedirs(os.path.dirname(env_dir), exist_ok=True)
    uv = _ensure_uv(progress)

    # uv downloads a standalone CPython if the pinned version isn't present, so
    # this works even with no Python installed on the machine.
    if managed_python() is None:
        # `only-managed` forces uv to use a standalone CPython it downloads,
        # rather than any (possibly incompatible / transient) system Python --
        # this is what makes the install reproducible on a machine with no
        # Python at all.
        _run([uv, "venv", env_dir, "--python", _MANAGED_PY,
              "--python-preference", "only-managed"], progress,
             "Creating local Python environment...")
    py = managed_python()
    if py is None:
        raise RuntimeError("Failed to create the local environment.")

    def _uv_pip_install(args, label):
        _run([uv, "pip", "install", "--python", py, *args], progress, label)

    # CPU torch (the cloud handles GPU). On Linux/Windows the default index ships
    # large CUDA wheels, so pin the CPU index there; macOS wheels are CPU/MPS.
    if _IS_WIN or sys.platform.startswith("linux"):
        _uv_pip_install(
            ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
            "Installing PyTorch (CPU)... this can take several minutes.")
    else:
        _uv_pip_install(
            ["torch"], "Installing PyTorch... this can take several minutes.")

    _uv_pip_install(["TotalSegmentator"], "Installing TotalSegmentator...")

    # Pre-download model weights so the first real run works offline. The console
    # script ships with TotalSegmentator; treat failure as non-fatal (weights
    # otherwise download on first use).
    dl = os.path.join(_venv_bin_dir(env_dir), _exe("totalseg_download_weights"))
    if os.path.exists(dl):
        try:
            _run([dl, "-t", "total_fast"], progress,
                 "Downloading segmentation model weights...")
        except RuntimeError:
            if progress:
                progress("Weight pre-download skipped; will download on first run.")

    if progress:
        progress("Local segmentation is ready.")
    ts = managed_ts_binary()
    if ts is None:
        raise RuntimeError("Install completed but TotalSegmentator was not found.")
    return ts
