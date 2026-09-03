"""Build the self-contained local-segmentation runtime bundled in the installer.

The full offline desktop build ships a prebuilt Python + PyTorch +
TotalSegmentator runtime *and* the pretrained model weights inside the app so a
user can segment locally with zero setup (no download, works offline).

RELOCATABILITY IS THE WHOLE POINT. The runtime is built on a CI runner but runs
from wherever the installer puts it ("C:\\Program Files\\Spine HU Tool", a
path with spaces, on a machine with no Python). A virtualenv can NEVER be
shipped this way: its ``pyvenv.cfg`` points at the base interpreter on the
build machine (which is not shipped) and its console-script executables embed
absolute build paths -- that was the v0.2.0 bug ("uv trampoline failed to
canonicalize script path" on every fresh install). So instead this script:

  1. installs a STANDALONE CPython (python-build-standalone via uv, relocatable
     by design) directly inside the bundle, and
  2. pip-installs torch + TotalSegmentator straight into that interpreter's own
     site-packages -- no venv, no ``pyvenv.cfg``, no console-script trampolines.

The app never runs any console script from the runtime; it invokes
``<bundled python> -c "...TotalSegmentator main..."`` (see
spine_hu_tool.segmentation.local_setup.ts_command), which no relocation can
break.

Usage (run in CI / on a build machine per OS -- NOT on a low-RAM dev laptop):

    python packaging/build_localseg_env.py --out packaging/localseg-bundle

Produces:
    <out>/localseg-env/       # standalone CPython + torch + TotalSegmentator
    <out>/totalseg-weights/   # pretrained weights (full-res + fast)

The PyInstaller spec / per-OS packagers copy these two directories next to the
frozen app (see packaging/spine_hu.spec). At runtime the app discovers them via
spine_hu_tool.segmentation.local_setup.bundled_env_dir / bundled_python.

IMPORTANT: this downloads/installs multi-GB packages and weights and must run on
a machine that can handle it. It never runs segmentation inference itself.
"""
from __future__ import annotations
import argparse
import os
import sys

# Make the package importable when run from a source checkout.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spine_hu_tool.segmentation import local_setup as ls  # noqa: E402


def _progress(msg: str) -> None:
    print(f"[build-localseg] {msg}", flush=True)


def _make_relocatable(env_dir: str) -> None:
    """Strip uv-store artifacts that would break or block the shipped runtime.

    - uv creates a version-less ``cpython-3.11-<plat>`` alias as an ABSOLUTE
      symlink to the real install: dangling after relocation -> remove.
    - uv marks its pythons ``EXTERNALLY-MANAGED`` (PEP 668), which blocks
      ensurepip/pip. This runtime is exclusively ours, so remove the marker.
    - drop uv's store bookkeeping (.lock, .temp, .gitignore).
    """
    for entry in os.listdir(env_dir):
        p = os.path.join(env_dir, entry)
        if os.path.islink(p):
            os.remove(p)
        elif entry in (".lock", ".gitignore"):
            os.remove(p)
        elif entry == ".temp":
            import shutil
            shutil.rmtree(p, ignore_errors=True)
    for root, _dirs, files in os.walk(env_dir):
        if "EXTERNALLY-MANAGED" in files:
            os.remove(os.path.join(root, "EXTERNALLY-MANAGED"))


def provision_standalone_runtime(env_dir: str, *, weights_dir: str,
                                 weight_tasks=("total", "total_fast"),
                                 progress=None,
                                 weights_fatal: bool = True) -> str:
    """Install a relocatable standalone CPython + ML stack into ``env_dir``.

    Returns the bundled interpreter path. Unlike
    :func:`local_setup.provision_env` (a venv for the in-place managed env),
    everything here lives inside ``env_dir`` itself and survives being copied
    to any path on any machine.
    """
    uv = ls._ensure_uv(progress)
    os.makedirs(env_dir, exist_ok=True)

    # 1. Standalone CPython (python-build-standalone) extracted INTO the bundle.
    #    UV_PYTHON_INSTALL_DIR redirects uv's managed-python store to env_dir, so
    #    the interpreter lands at <env_dir>/cpython-<ver>-<platform>/. --no-bin
    #    stops uv from also dropping shims into ~/.local/bin.
    ls._run([uv, "python", "install", ls._MANAGED_PY, "--no-bin"], progress,
            "Installing standalone CPython into the bundle...",
            env=ls.child_env({"UV_PYTHON_INSTALL_DIR": env_dir}))
    _make_relocatable(env_dir)
    py = ls.find_runtime_python(env_dir)
    if py is None:
        raise RuntimeError(f"No interpreter found under {env_dir} after "
                           "'uv python install'.")

    # 2. Packages straight into the standalone interpreter's site-packages.
    #    Plain pip (bootstrapped by the bundled ensurepip) keeps uv's venv
    #    machinery entirely out of the shipped artifact.
    ls._run([py, "-m", "ensurepip", "--upgrade"], progress,
            "Bootstrapping pip in the bundled Python...")
    pip = [py, "-m", "pip", "install", "--no-warn-script-location"]
    # CPU torch (the cloud handles GPU). On Linux/Windows the default index
    # ships large CUDA wheels, so pin the CPU index there; macOS wheels are
    # CPU/MPS already.
    if sys.platform.startswith(("win", "linux")):
        ls._run([*pip, "torch", "--index-url",
                 "https://download.pytorch.org/whl/cpu"], progress,
                "Installing PyTorch (CPU)... this can take several minutes.")
    else:
        ls._run([*pip, "torch"], progress,
                "Installing PyTorch... this can take several minutes.")
    ls._run([*pip, "TotalSegmentator"], progress,
            "Installing TotalSegmentator...")

    # 3. Pretrained weights, fetched through the bundled interpreter itself.
    ls._download_weights(py, weights_dir, weight_tasks, progress, weights_fatal)
    return py


def _slim_env(env_dir: str) -> None:
    """Trim bundle bloat and, critically, the deeply-nested vendored license
    trees that overflow Windows' 260-char MAX_PATH when the installer is built.

    torch ships ``*.dist-info/licenses/third_party/.../LICENSE`` paths hundreds
    of characters deep; Inno Setup can't read them on Windows ("The system
    cannot find the path specified"). We drop those nested ``third_party``
    license subtrees (torch's own top-level LICENSE stays) plus ``__pycache__``
    and stray ``.pyc`` files. This does not touch importable code, so the
    runtime is unaffected -- it just makes the tree shallower and smaller.
    """
    import shutil

    removed = {"licenses": 0, "pycache": 0}
    for root, dirs, files in os.walk(env_dir, topdown=True):
        base = os.path.basename(root)
        # prune nested vendored license trees inside any *.dist-info/licenses
        if base == "third_party" and f"licenses{os.sep}" in (root + os.sep):
            parent = os.path.dirname(root)
            if ".dist-info" in parent or parent.endswith("licenses"):
                shutil.rmtree(root, ignore_errors=True)
                removed["licenses"] += 1
                dirs[:] = []
                continue
        if base == "__pycache__":
            shutil.rmtree(root, ignore_errors=True)
            removed["pycache"] += 1
            dirs[:] = []
            continue
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass
    _progress(f"Slimmed env: removed {removed['licenses']} nested license "
              f"tree(s), {removed['pycache']} __pycache__ dir(s).")


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

    _progress(f"Building relocatable local-seg runtime at {env_dir}")
    _progress(f"Weights -> {weights_dir} (tasks: {', '.join(args.tasks)})")
    py = provision_standalone_runtime(
        env_dir, weights_dir=weights_dir, weight_tasks=tuple(args.tasks),
        progress=_progress, weights_fatal=not args.best_effort_weights)
    _slim_env(env_dir)

    # In-situ health check (the relocation check runs in verify_build.py).
    ok, reason = ls.verify_runtime(py, use_cache=False)
    if not ok:
        _progress(f"ERROR: built runtime failed its health check:\n{reason}")
        return 1
    _progress(f"Done. Bundled Python: {py}")

    # Sanity: weights dir must be non-empty when weights are required.
    if not args.best_effort_weights:
        has_files = any(os.scandir(weights_dir)) if os.path.isdir(weights_dir) else False
        if not has_files:
            _progress("ERROR: weights directory is empty after download.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
