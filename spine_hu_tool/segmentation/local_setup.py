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
import re
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


_health_cache: Optional[bool] = None


def invalidate_health_cache() -> None:
    global _health_cache
    _health_cache = None


def env_python_ok() -> bool:
    """Quick health check: can the managed env's Python actually start?

    File existence is NOT enough: a leftover env whose base interpreter is
    gone or unreachable (e.g. uv's roaming minor-version junction became
    untrusted, or the roaming uv dir was cleaned up) still has python.exe and
    TotalSegmentator.exe on disk, but every launch fails ("uv trampoline
    failed to spawn Python child process: entity not found"). Spawning the
    interpreter once (<1 s) catches that whole class of breakage. The result
    is cached per process; (re)installing invalidates it.
    """
    global _health_cache
    if _health_cache is not None:
        return _health_cache
    py = managed_python()
    if py is None:
        return False              # not cached: the env may appear later
    try:
        r = subprocess.run([py, "-c", "pass"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=60)
        _health_cache = (r.returncode == 0)
    except (OSError, subprocess.SubprocessError):
        _health_cache = False
    return _health_cache


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


def _uv_dirs() -> dict:
    """uv state dirs pinned inside our per-user LOCAL data dir.

    By default uv keeps its managed Pythons in the *roaming* profile
    (``%APPDATA%\\uv``) and reaches them through a minor-version directory
    junction. On managed / OneDrive-synced Windows machines the filesystem
    filter marks that junction untrusted, after which every traversal fails
    with os error 448 ("The path cannot be traversed because it contains an
    untrusted mount point") -- this broke local setup on a pilot machine.
    Pinning uv's data/cache/python dirs into our own LOCALAPPDATA folder
    (never roamed or cloud-synced) sidesteps the poisoned location entirely.
    """
    root = os.path.join(_user_data_root(), "uv")
    return {
        "UV_DATA_DIR": root,
        "UV_CACHE_DIR": os.path.join(root, "cache"),
        "UV_PYTHON_INSTALL_DIR": os.path.join(root, "python"),
    }


def _uv_env() -> dict:
    env = dict(os.environ)
    env.update(_uv_dirs())
    return env


def _find_installed_python() -> Optional[str]:
    """Locate the real (fully-versioned) CPython that uv installed.

    uv lays the interpreter out as ``<dir>/cpython-3.11.9-<plat>/...`` and adds
    a ``cpython-3.11-<plat>`` minor-version *junction* beside it. That junction
    is exactly what untrusted-mount-point hardening refuses to traverse, so we
    resolve the real directory by name (three-component version) and never go
    through the link.
    """
    install_dir = _uv_dirs()["UV_PYTHON_INSTALL_DIR"]
    if not os.path.isdir(install_dir):
        return None
    pat = re.compile(rf"^cpython-{re.escape(_MANAGED_PY)}\.\d+")
    for entry in sorted(os.listdir(install_dir), reverse=True):
        if not pat.match(entry):
            continue
        base = os.path.join(install_dir, entry)
        candidates = ((os.path.join(base, "python.exe"),) if _IS_WIN else
                      (os.path.join(base, "bin", "python3"),
                       os.path.join(base, "bin", "python")))
        for cand in candidates:
            if os.path.exists(cand):
                return cand
    return None


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
    # Self-heal: an env that exists but is not ready is broken or half
    # installed (orphaned base interpreter, interrupted install, ...).
    # Re-running pip against it would fail with the same opaque errors the
    # user is trying to escape, so rebuild it from scratch.
    if os.path.isdir(env_dir):
        if progress:
            progress("Removing the previous (broken) local environment...")
        shutil.rmtree(env_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(env_dir), exist_ok=True)
    uv = _ensure_uv(progress)

    # uv downloads a standalone CPython if the pinned version isn't present, so
    # this works even with no Python installed on the machine.
    if managed_python() is None:
        # Download a standalone CPython into our own local, non-roaming dir
        # (see _uv_dirs). On 448-hardened machines uv can exit nonzero here
        # solely because it failed to create its minor-version junction; the
        # interpreter itself lands fine, so tolerate the failure as long as a
        # usable interpreter exists afterwards.
        if _find_installed_python() is None:
            try:
                _run([uv, "python", "install", _MANAGED_PY], progress,
                     "Downloading a private Python runtime...", env=_uv_env())
            except RuntimeError:
                if _find_installed_python() is None:
                    raise
        base_py = _find_installed_python()
        if base_py is None:
            raise RuntimeError("Could not install the private Python runtime.")
        # Create the env from the explicit interpreter path: uv then treats it
        # as an external interpreter and never resolves it through the managed
        # minor-version junction that hardened machines refuse to traverse.
        _run([uv, "venv", env_dir, "--python", base_py], progress,
             "Creating local Python environment...", env=_uv_env())
    py = managed_python()
    if py is None:
        raise RuntimeError("Failed to create the local environment.")

    def _uv_pip_install(args, label):
        _run([uv, "pip", "install", "--python", py, *args], progress, label,
             env=_uv_env())

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
    invalidate_health_cache()
    ts = managed_ts_binary()
    if ts is None:
        raise RuntimeError("Install completed but TotalSegmentator was not found.")
    return ts
