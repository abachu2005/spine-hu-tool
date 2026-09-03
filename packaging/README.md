# `packaging/` — installers & distribution

How the tool gets onto a physician's or labmate's machine. The goal: a
**double-clickable native installer for every OS, no Python required**, small
enough to download over the web.

| Path | What it is |
|------|-----------|
| `spine_hu.spec`, `spine_hu_app.py` | PyInstaller build spec + app entry. |
| `make_icons.py`, `icons/` | Generate `.icns` / `.ico` / `.png` app icons. |
| `macos/build_dmg.sh` | Build the macOS `.dmg`. |
| `windows/installer.iss` | Inno Setup script → Windows setup wizard. |
| `linux/build_appimage.sh` | Build the Linux AppImage. |
| `download/index.html`, `version.json` | The public download page + version metadata. |
| `upload_to_gcs.sh` | Publish page + installers to the public GCS bucket. |
| `sync_version.py` | Propagate one version number to every file that ships one. |
| `SIGNING.md` | Code-signing notes. |

## One version number

`spine_hu_tool.__version__` is the source of truth, because it is what every
export stamps into `reproducibility.json` as `tool_version` — the field that
ties a measurement back to the code that produced it. A build that advertises
one version while stamping another makes that field worthless.

`pyproject.toml` and `spine_hu.spec` read the attribute directly. Inno Setup
cannot (it has no way to run Python) and the download page's `version.json` is
served as a static file, so those two carry a copy that `sync_version.py`
rewrites:

```bash
python packaging/sync_version.py --set 0.1.4   # cut a release
python packaging/sync_version.py --check       # what CI runs
```

CI runs `--check` before building anything, and on a tag push it additionally
requires `__version__` to equal the tag, so `v0.1.4` cannot ship binaries whose
exports identify themselves as `0.1.3`.

## Why lean, cloud-segmenting installers

If the installer bundled PyTorch + TotalSegmentator + weights it would be
multi-GB and painful to distribute. Instead the packaged app ships **without the
ML stack** (~130 MB) and **segments on the cloud by default**. Anyone who wants
local, no-upload processing installs the runtime **on demand** (see
`spine_hu_tool/segmentation/local_setup.py`), so the download stays small while
local processing is still one click away.

## Why native installers per OS

Labmates run macOS, Windows, and Linux and are not developers. A per-OS native
installer (`.dmg`, Inno Setup wizard, AppImage) is the lowest-friction path — no
terminal, no `pip`, no Python version to match. First-launch friction from
unsigned apps is documented in the top-level README (right-click → Open on macOS,
"More info → Run anyway" on Windows SmartScreen).

## Why GCS distribution (public-but-unlisted)

The code lives in a **private** GitHub repo, but installers are published to a
**public-but-unlisted GCS bucket** so labmates can download without a GitHub
login or credentials. `upload_to_gcs.sh` publishes the download page and version
metadata with `no-cache` (so updates show immediately) and installers with a long
cache (immutable per release). CI does the same automatically on a tag push.

## Distribution flow

```mermaid
flowchart LR
    tag["git tag / push"] --> check["Version consistency check"]
    check --> ci["CI (GitHub Actions)"]
    ci --> mac["macOS .dmg"]
    ci --> win["Windows setup.exe"]
    ci --> lin["Linux AppImage"]
    mac --> gcs["Public GCS bucket"]
    win --> gcs
    lin --> gcs
    gcs --> page["Download page (index.html)"]
    page --> user["Labmate downloads & runs"]
```
