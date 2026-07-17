"""Copy the prebuilt local-seg runtime + weights into a packaged app.

Run AFTER PyInstaller and BEFORE the OS packager wraps the app (dmg / setup.exe
/ AppImage). Copying here -- rather than via PyInstaller ``datas`` -- preserves
executable bits on the bundled Python / TotalSegmentator binaries (PyInstaller
can strip perms on data files) and keeps PyInstaller from analyzing/rewriting
the multi-GB env.

The runtime lands where ``spine_hu_tool.segmentation.local_setup`` discovers it:
  - macOS ``.app``  -> ``Contents/Resources/{localseg-env,totalseg-weights}``
  - onedir (win/linux) -> ``<app_dir>/{localseg-env,totalseg-weights}``
    (next to the executable; discovery also checks ``_internal``)

Usage:
    python packaging/bundle_localseg.py \
        --bundle packaging/localseg-bundle \
        --app "packaging/dist/Spine HU Tool.app"
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys


def _dest_root(app_path: str) -> str:
    """Where to place the runtime inside the given app."""
    if app_path.endswith(".app") or os.path.isdir(os.path.join(app_path, "Contents")):
        return os.path.join(app_path, "Contents", "Resources")
    return app_path


def install_into(app_path: str, bundle_dir: str) -> None:
    dest_root = _dest_root(app_path)
    os.makedirs(dest_root, exist_ok=True)
    for name in ("localseg-env", "totalseg-weights"):
        src = os.path.join(bundle_dir, name)
        if not os.path.isdir(src):
            raise SystemExit(f"error: expected '{src}' from build_localseg_env.py")
        dst = os.path.join(dest_root, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        # copytree preserves file permissions (exec bits) via copy2.
        shutil.copytree(src, dst, symlinks=True)
        print(f"[bundle-localseg] {src} -> {dst}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="packaging/localseg-bundle",
                    help="dir produced by build_localseg_env.py")
    ap.add_argument("--app", required=True,
                    help="path to the .app (macOS) or onedir app directory")
    args = ap.parse_args(argv)
    install_into(os.path.abspath(args.app), os.path.abspath(args.bundle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
