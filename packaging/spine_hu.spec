# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Spine HU Tool desktop client.

Builds a onedir bundle named "Spine HU Tool". On macOS it is wrapped into a
.app via BUNDLE. Heavy ML deps (torch/TotalSegmentator/nnU-Net) are explicitly
excluded -- segmentation runs on the cloud service -- so the bundle stays in
the hundreds-of-MB range.

Usage (from repo root):
    pyinstaller packaging/spine_hu.spec --noconfirm
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# --- paths ---------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ENTRY = os.path.join(SPECPATH, "spine_hu_app.py")
ICON_DIR = os.path.join(SPECPATH, "icons")

APP_NAME = "Spine HU Tool"

if sys.platform == "darwin":
    icon_file = os.path.join(ICON_DIR, "spine_hu.icns")
elif sys.platform.startswith("win"):
    icon_file = os.path.join(ICON_DIR, "spine_hu.ico")
else:
    icon_file = os.path.join(ICON_DIR, "spine_hu.png")
if not os.path.exists(icon_file):
    icon_file = None

# --- collect data + hidden imports for native/plugin-heavy deps ----------
datas = [(ICON_DIR, "icons")]
binaries = []
hiddenimports = []

for pkg in ("SimpleITK", "pydicom", "pyqtgraph"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# matplotlib: ship data, but only the Agg/Qt backends are needed.
mpl_d, mpl_b, mpl_h = collect_all("matplotlib")
datas += mpl_d
binaries += mpl_b
hiddenimports += mpl_h

hiddenimports += collect_submodules("scipy")
hiddenimports += [
    "PySide6.QtSvg",        # icon / vector rendering
    "PySide6.QtPrintSupport",
]

# Keep the bundle lean: segmentation is offloaded to the cloud, so the giant
# ML stack must never be pulled in.
excludes = [
    "torch", "torchvision", "totalsegmentator", "nnunetv2", "nnunet",
    "tensorflow", "tkinter", "PyQt5", "PyQt6", "IPython", "jupyter",
    "pytest", "fastapi", "uvicorn", "starlette",
]

block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon_file,
        bundle_identifier="com.spinehu.tool",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Pilot / research use only.",
        },
    )
