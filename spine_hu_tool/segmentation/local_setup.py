"""On-demand, self-contained setup of the local segmentation runtime.

The packaged desktop app ships lean (no PyTorch / TotalSegmentator) and segments
on the cloud by default. A user who wants to run segmentation on their own
machine can install the heavy ML stack once into a *managed* environment that
lives in a user-writable data directory (outside the read-only app bundle). The
app then runs TotalSegmentator as a subprocess of that environment's Python
(never its console-script executables -- see the trampoline note below).

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
from typing import Callable, Optional, Tuple

from ..errors import UserFacingError

ProgressCb = Optional[Callable[[str], None]]

_IS_WIN = sys.platform.startswith("win")

# Pinned Python for the managed env: a version with known torch / TotalSegmentator
# wheels (uv downloads a standalone build of it -- no system Python needed).
_MANAGED_PY = "3.11"

# Env vars PyInstaller injects into the frozen process so the dynamic loader and
# Python point at the bundled runtime. If these leak into a child process (our
# bundled / managed Python + TotalSegmentator) the child loads the frozen app's
# libraries instead of its own -- the classic "runs from source but the packaged
# app's local segmentation silently fails / crashes" bug. PyInstaller stashes the
# pre-launch values in ``<VAR>_ORIG``; restore those when present, else drop the
# variable entirely so the child gets a clean, self-contained environment.
_LEAKY_ENV_VARS = (
    "LD_LIBRARY_PATH", "LD_PRELOAD",
    "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_INSERT_LIBRARIES",
    "PYTHONHOME", "PYTHONPATH",
)


def child_env(extra: Optional[dict] = None) -> dict:
    """A clean environment for launching the bundled/managed Python or TS.

    Strips PyInstaller's injected loader/Python vars (restoring the originals it
    saved in ``<VAR>_ORIG`` when available), so the child never inherits the
    frozen app's bundled libraries. ``extra`` values are applied last; ``None``
    values in ``extra`` are ignored.
    """
    env = dict(os.environ)
    for var in _LEAKY_ENV_VARS:
        orig = env.pop(var + "_ORIG", None)
        if orig:
            env[var] = orig
        else:
            env.pop(var, None)
    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env


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


# --- console-script-free entrypoints ------------------------------------------
#
# NEVER invoke the env's console-script executables (TotalSegmentator.exe etc.):
# on Windows those are uv/pip trampolines that embed absolute build paths and
# break when the env is relocated (the shipped v0.2.0 bundle) or when the path
# contains spaces ("C:\Program Files\...", a known uv trampoline bug). Running
# the interpreter itself with a -c shim is immune to both, and to missing
# __main__ guards.

_TS_ENTRY = ("import sys; from totalsegmentator.bin.TotalSegmentator "
             "import main; sys.exit(main())")
_DL_ENTRY = ("import sys; from totalsegmentator.bin.totalseg_download_weights "
             "import main; sys.exit(main())")


def ts_command(python: str, *ts_args: str) -> list:
    """Command that runs TotalSegmentator through the given interpreter."""
    return [python, "-c", _TS_ENTRY, *ts_args]


def weights_download_command(python: str, task: str) -> list:
    """Command that pre-downloads a weights task through the interpreter."""
    return [python, "-c", _DL_ENTRY, "-t", task]


# --- bundled runtime (shipped inside the installer) ---------------------------
#
# The full offline build ships a prebuilt segmentation env and pretrained model
# weights *inside* the app so local segmentation works with zero setup. These
# live next to the frozen app; the exact directory depends on how PyInstaller
# lays out the bundle per OS, so probe a list of candidates rather than assume
# one path. Env var overrides let the build/tests point at an arbitrary path.

def _bundle_roots() -> list:
    """Candidate directories that may contain the bundled runtime/weights."""
    roots: list = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(meipass)
        # macOS .app BUNDLE: data lands in Contents/Frameworks while _MEIPASS may
        # be Contents/Frameworks or Contents/MacOS; the app icon/resources sit in
        # Contents/Resources. Cover the siblings.
        parent = os.path.dirname(meipass)
        roots.append(os.path.join(parent, "Resources"))
        roots.append(parent)
    exe_dir = os.path.dirname(sys.executable)
    roots.append(exe_dir)
    roots.append(os.path.join(exe_dir, "_internal"))
    # macOS: Contents/MacOS/<exe> -> Contents/Resources
    roots.append(os.path.join(exe_dir, "..", "Resources"))
    # de-dupe while preserving order
    seen = set()
    out = []
    for r in roots:
        ap = os.path.abspath(r)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def bundled_env_dir() -> Optional[str]:
    """Directory of the runtime bundled with the installer, or None.

    Overridable with ``SPINE_HU_BUNDLED_ENV`` (used by the build/verify script).
    """
    override = os.environ.get("SPINE_HU_BUNDLED_ENV")
    if override:
        return override if os.path.isdir(override) else None
    for root in _bundle_roots():
        cand = os.path.join(root, "localseg-env")
        if os.path.isdir(cand):
            return cand
    return None


def find_runtime_python(root: Optional[str]) -> Optional[str]:
    """Locate a CPython interpreter inside an installed runtime directory.

    Handles every layout we ship or shipped:
      - standalone CPython installed into the bundle by uv
        (``<root>/cpython-3.11.x-<platform>/{python.exe | bin/python3.x}``)
      - a flat standalone build (``<root>/{python.exe | bin/python3}``)
      - a legacy venv (``<root>/{Scripts | bin}/python[.exe]``)
    """
    if not root or not os.path.isdir(root):
        return None
    bases = [root]
    try:
        # Skip symlinked cpython-* aliases: uv creates a version-less alias as
        # an ABSOLUTE symlink, which dangles once the bundle is relocated.
        bases += sorted(os.path.join(root, d) for d in os.listdir(root)
                        if d.startswith("cpython-")
                        and os.path.isdir(os.path.join(root, d))
                        and not os.path.islink(os.path.join(root, d)))
    except OSError:
        return None
    for base in bases:
        subdirs = ("", "Scripts") if _IS_WIN else ("bin", "")
        for sub in subdirs:
            d = os.path.join(base, sub) if sub else base
            if _IS_WIN:
                cand = os.path.join(d, "python.exe")
                if os.path.isfile(cand):
                    return cand
            else:
                try:
                    names = sorted(os.listdir(d))
                except OSError:
                    continue
                for n in names:
                    if n == "python3" or n == "python" or (
                            n.startswith("python3.") and not n.endswith("-config")):
                        cand = os.path.join(d, n)
                        if os.path.isfile(cand) and os.access(cand, os.X_OK):
                            return cand
    return None


def bundled_python() -> Optional[str]:
    """Interpreter of the runtime bundled with the installer, or None."""
    return find_runtime_python(bundled_env_dir())


def bundled_weights_dir() -> Optional[str]:
    """Directory of the pretrained weights bundled with the installer, or None.

    Overridable with ``SPINE_HU_BUNDLED_WEIGHTS``.
    """
    override = os.environ.get("SPINE_HU_BUNDLED_WEIGHTS")
    if override:
        return override if os.path.isdir(override) else None
    for root in _bundle_roots():
        cand = os.path.join(root, "totalseg-weights")
        if os.path.isdir(cand):
            return cand
    return None


def weights_home() -> Optional[str]:
    """Weights directory to expose to TotalSegmentator (``TOTALSEG_HOME_DIR``).

    Prefer an explicit override, then the bundled weights, then the managed
    env's own weights dir if it exists. ``None`` means "let TotalSegmentator use
    its default" (which downloads on first use).
    """
    override = os.environ.get("SPINE_HU_WEIGHTS_HOME")
    if override:
        return override
    bundled = bundled_weights_dir()
    if bundled:
        return bundled
    return None


def is_ready() -> bool:
    """True if the managed local-segmentation runtime is fully installed."""
    ok, _ = verify_runtime(managed_python())
    return ok


# --- runtime health check + resolution -----------------------------------------

# verify_runtime results per interpreter path; a health check launches a Python
# and imports torch (seconds), so it runs once per process, not once per run.
_VERIFY_CACHE: dict = {}


def clear_verify_cache() -> None:
    _VERIFY_CACHE.clear()


def verify_runtime(python: Optional[str], *, use_cache: bool = True,
                   timeout: float = 180) -> Tuple[bool, str]:
    """Preflight health check: can ``python`` import torch + totalsegmentator?

    Returns ``(ok, reason)`` where ``reason`` is the captured import error /
    launch failure when not ok. Run with the sanitized :func:`child_env` -- the
    same environment real runs use -- so it catches the frozen-app env-leak
    class of failure too.
    """
    if not python or not os.path.exists(python):
        return False, f"interpreter not found ({python!r})"
    key = os.path.abspath(python)
    if use_cache and key in _VERIFY_CACHE:
        return _VERIFY_CACHE[key]
    try:
        r = subprocess.run([python, "-c", "import torch, totalsegmentator"],
                           capture_output=True, text=True, timeout=timeout,
                           env=child_env())
        ok = r.returncode == 0
        reason = "" if ok else (r.stderr or r.stdout or "").strip()[-2000:]
    except (OSError, subprocess.SubprocessError) as e:
        ok, reason = False, f"failed to launch the runtime: {e}"
    result = (ok, reason)
    _VERIFY_CACHE[key] = result
    return result


def runtime_candidates() -> list:
    """``(source, python)`` pairs in preference order (existence-checked only).

    Order: the runtime bundled inside the installer, then the user-data managed
    env (which is also the repair target, so a repaired env can take over from a
    broken bundled one), then the host interpreter for source installs with the
    ``local-seg`` extra.
    """
    out = []
    b = bundled_python()
    if b:
        out.append(("bundled", b))
    m = managed_python()
    if m:
        out.append(("managed", m))
    try:
        import importlib.util
        if importlib.util.find_spec("totalsegmentator") is not None:
            out.append(("host", sys.executable))
    except Exception:
        pass
    return out


def _active_overrides() -> list:
    return [f"{v}={os.environ[v]}"
            for v in ("SPINE_HU_BUNDLED_ENV", "SPINE_HU_BUNDLED_WEIGHTS",
                      "SPINE_HU_LOCALSEG_ENV", "SPINE_HU_WEIGHTS_HOME")
            if os.environ.get(v)]


def resolve_runtime() -> Tuple[str, str]:
    """Return ``(python, source)`` of the first healthy local-seg runtime.

    Tries every candidate (bundled -> managed -> host) and health-checks each.
    Raises :class:`UserFacingError` naming every resolved path, its failure
    reason, and any active ``SPINE_HU_*`` overrides when none is usable.
    """
    failures = []
    for source, py in runtime_candidates():
        ok, reason = verify_runtime(py)
        if ok:
            return py, source
        failures.append(f"[{source}] {py}:\n{reason}")
    detail = ("\n\n".join(failures)
              or "No local segmentation runtime found (no bundled runtime, no "
                 "managed env, and TotalSegmentator is not importable).")
    overrides = _active_overrides()
    if overrides:
        detail += "\n\nActive environment overrides: " + ", ".join(overrides)
    raise UserFacingError(
        title="Local segmentation isn't working",
        message="Local segmentation on this computer isn't working.",
        remedy="Click 'Repair local segmentation' to reinstall it (needs "
               "internet, a few minutes), or click 'Use cloud instead' to "
               "continue right now.",
        detail=detail, kind="local-runtime-broken")


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
    """Sanitized child environment with the pinned uv dirs applied."""
    return child_env(_uv_dirs())


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
    # Always launch with a sanitized environment so a frozen host app's bundled
    # loader/Python vars never leak into uv / the managed Python.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env=env or child_env())
    last = ""
    for line in proc.stdout:                       # stream so the UI stays live
        last = line.strip()
        if progress and last:
            progress(last[-160:])
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{last}")


def _venv_python(env_dir: str) -> Optional[str]:
    cand = os.path.join(_venv_bin_dir(env_dir), _exe("python"))
    return cand if os.path.exists(cand) else None


def _download_weights(py: str, weights_dir: Optional[str], weight_tasks,
                      progress: ProgressCb, weights_fatal: bool) -> None:
    """Pre-download model weights through the env's interpreter (no console
    script -- see the trampoline note above) so the first real run is offline."""
    dl_env = child_env({"TOTALSEG_HOME_DIR": weights_dir,
                        "TOTALSEG_WEIGHTS_PATH": weights_dir})
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
    for task in weight_tasks:
        try:
            _run(weights_download_command(py, task), progress,
                 f"Downloading segmentation model weights ({task})...",
                 env=dl_env)
        except RuntimeError:
            if weights_fatal:
                raise
            if progress:
                progress(f"Weight pre-download ({task}) skipped; will "
                         "download on first run.")


def provision_env(env_dir: str, *, weights_dir: Optional[str] = None,
                  weight_tasks=("total_fast",), progress: ProgressCb = None,
                  weights_fatal: bool = False) -> str:
    """Install a self-contained torch + TotalSegmentator venv into ``env_dir``.

    Used by the on-demand user install (:func:`setup_local_seg`), where the env
    is created IN PLACE on the user's machine and never moves, so a venv is
    fine. (The installer bundle must NOT use this: a venv references the
    CPython that created it and breaks when relocated -- the v0.2.0 bug. The
    bundle uses ``packaging/build_localseg_env.py``'s standalone runtime.)

    Bootstraps ``uv`` (which downloads a standalone CPython -- no system Python
    needed), installs CPU PyTorch + TotalSegmentator, and pre-downloads the
    given weight ``weight_tasks``. When ``weights_dir`` is given, weights are
    downloaded there (via ``TOTALSEG_HOME_DIR``) so they can live separately
    from the env. Returns the env's Python interpreter path.

    ``weights_fatal`` makes a failed weight download raise; the on-demand path
    treats it as best-effort since weights otherwise download on first use.
    """
    os.makedirs(os.path.dirname(env_dir) or ".", exist_ok=True)
    uv = _ensure_uv(progress)

    if _venv_python(env_dir) is None:
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
    py = _venv_python(env_dir)
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

    _download_weights(py, weights_dir, weight_tasks, progress, weights_fatal)
    clear_verify_cache()               # a fresh env invalidates cached health
    return py


def setup_local_seg(progress: ProgressCb = None) -> str:
    """Provision the managed env and install the local segmentation stack.

    Self-contained: bootstraps `uv`, which downloads a standalone Python, then
    installs a CPU PyTorch + TotalSegmentator and pre-downloads the model
    weights. Returns the managed env's Python path. Safe to re-run.
    """
    if is_ready():
        if progress:
            progress("Local segmentation already installed.")
        return managed_python()  # type: ignore[return-value]

    env_dir = managed_env_dir()
    # Self-heal: an env that exists but is not ready is broken or half
    # installed (orphaned base interpreter, interrupted install, ...).
    # Re-running pip against it would fail with the same opaque errors the
    # user is trying to escape, so rebuild it from scratch.
    if os.path.isdir(env_dir):
        if progress:
            progress("Removing the previous (broken) local environment...")
        shutil.rmtree(env_dir, ignore_errors=True)
        clear_verify_cache()

    py = provision_env(env_dir, weight_tasks=("total_fast",),
                       progress=progress)
    ok, reason = verify_runtime(py, use_cache=False)
    if not ok:
        raise UserFacingError(
            title="Local setup failed",
            message="The local segmentation runtime was installed but failed "
                    "its health check.",
            remedy="Try 'Set up local segmentation' again; if it keeps "
                   "failing, use cloud segmentation and send us the log file.",
            detail=f"{py}:\n{reason}", kind="setup-failed")
    if progress:
        progress("Local segmentation is ready.")
    return py


def classify_setup_error(exc: BaseException) -> UserFacingError:
    """Turn a raw setup/repair failure into an actionable user-facing error.

    Distinguishes the two failure modes a user can actually fix themselves --
    no internet and no disk space -- from everything else, instead of showing
    raw pip/uv output as the headline.
    """
    if isinstance(exc, UserFacingError):
        return exc
    s = str(exc)
    low = s.lower()
    if any(k in low for k in (
            "no space left", "disk full", "errno 28", "not enough space",
            "insufficient disk", "disk quota")):
        return UserFacingError(
            title="Not enough disk space",
            message="There isn't enough free disk space to install local "
                    "segmentation (it needs about 3 GB free).",
            remedy="Free up disk space, then click 'Set up local "
                   "segmentation' again.",
            detail=s, kind="setup-failed")
    if any(k in low for k in (
            "getaddrinfo", "name resolution", "temporary failure",
            "connection", "network", "timed out", "timeout", "unreachable",
            "ssl", "proxy", "url error", "urlopen", "http error 5")):
        return UserFacingError(
            title="Couldn't download the components",
            message="Local segmentation setup needs to download its "
                    "components, but the download failed -- this usually "
                    "means no internet connection.",
            remedy="Connect to the internet and click 'Set up local "
                   "segmentation' again. On a hospital network, a firewall "
                   "or proxy may be blocking downloads.",
            detail=s, kind="setup-failed")
    return UserFacingError(
        title="Local setup failed",
        message="Local segmentation could not be installed on this computer.",
        remedy="Try again; if it keeps failing, use cloud segmentation "
               "(uncheck 'Run segmentation on this computer') and send us "
               "the log file.",
        detail=s, kind="setup-failed")


def repair_local_seg(progress: ProgressCb = None) -> str:
    """Delete the managed env and re-provision it from scratch.

    The bundled runtime lives in the read-only app directory, so 'repair' means
    rebuilding the user-data managed env -- which :func:`runtime_candidates`
    prefers over a bundled runtime that fails its health check. Returns the
    fresh env's Python path.
    """
    env_dir = managed_env_dir()
    if os.path.isdir(env_dir):
        if progress:
            progress("Removing the existing local environment...")
        shutil.rmtree(env_dir, ignore_errors=True)
    clear_verify_cache()
    return setup_local_seg(progress=progress)
