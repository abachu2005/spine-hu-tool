# Offline build — cross-OS test checklist

The full offline build ships the segmentation runtime (PyTorch +
TotalSegmentator) and pretrained weights **inside** the installer so a user can
segment locally with **zero setup**, fully offline. Because the developer laptop
cannot run local segmentation (it will crash on a full-res run), the real
end-to-end validation happens in CI and on the per-OS test machines below.

## What CI does automatically (on a `v*` tag or manual dispatch)

1. `packaging/build_localseg_env.py` builds the self-contained runtime + weights
   (full-res `total` + fast `total_fast`).
2. PyInstaller builds the lean frozen app.
3. `packaging/bundle_localseg.py` copies the runtime into the app (macOS
   adhoc-signs afterward so nested binaries launch unsigned).
4. The OS packager wraps it (`.dmg` / setup `.exe` / AppImage).
5. Installers publish to the **GCS bucket** (authoritative; GitHub Releases can't
   hold the multi-GB assets).

## Automated build verification (run on a machine with enough RAM)

```bash
# import-only (light): confirms bundled runtime + weights are present & importable
python packaging/verify_build.py --app "packaging/dist/Spine HU Tool.app"

# full (heavy): also runs a tiny real offline segmentation — needs RAM
python packaging/verify_build.py --app "packaging/dist/Spine HU Tool" --run-seg
```

## Manual matrix — do this for each OS before announcing a release

For every target — **macOS (Apple Silicon)**, **macOS (Intel)**, **Windows**,
**Linux** — on a machine that does NOT have the dev tools installed:

1. Download the installer from the download page (not from a local build).
2. Install / open it, getting past the unsigned warning:
   - macOS: right-click the app → **Open** → **Open**; or
     `xattr -dr com.apple.quarantine "/Applications/Spine HU Tool.app"`.
   - Windows: **More info** → **Run anyway** (SmartScreen).
   - Linux: `chmod +x` the AppImage, then run it.
3. Turn OFF networking (airplane mode / pull the cable) to prove offline.
4. Open a de-identified DICOM study.
5. Check **Run segmentation on this computer (no upload)**.
6. Leave resolution on **Full resolution** on a strong machine; on an 8 GB
   laptop confirm the **low-memory warning** appears and "Use fast instead"
   works.
7. Click **Analyze** and confirm it segments and shows the review screen with
   measurements — with no network.
8. Repeat once in **Fast (3 mm)** mode.

## Notes on the fixed failure mode

The previous "local setup didn't work" was primarily frozen-app **environment
leakage**: PyInstaller injects `DYLD_*` / `LD_LIBRARY_PATH` / `PYTHONHOME` /
`PYTHONPATH` into the app process, and those leaked into the child Python /
TotalSegmentator, making it load the wrong libraries. The runtime now launches
every subprocess through `local_setup.child_env()`, which strips those vars
(restoring PyInstaller's saved `*_ORIG` values when present). `verify_build.py`
exercises exactly this path.
