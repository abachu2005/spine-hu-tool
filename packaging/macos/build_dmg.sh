#!/usr/bin/env bash
# Build the macOS .dmg from the PyInstaller .app bundle.
#
# Prereqs:
#   pip install pyinstaller
#   brew install create-dmg   (falls back to hdiutil if not installed)
#
# Usage (from repo root):
#   bash packaging/macos/build_dmg.sh
set -euo pipefail

APP_NAME="Spine HU Tool"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST="$ROOT/packaging/dist"
APP="$DIST/$APP_NAME.app"
DMG="$DIST/$APP_NAME.dmg"
VOL_ICON="$ROOT/packaging/icons/spine_hu.icns"

# 1. build the .app if missing
if [[ ! -d "$APP" ]]; then
  echo "==> Building app bundle with PyInstaller"
  pyinstaller "$ROOT/packaging/spine_hu.spec" --noconfirm \
    --distpath "$DIST" --workpath "$ROOT/packaging/build"
fi

# 1b. FULL OFFLINE build: copy the prebuilt local-seg runtime + weights into the
# .app so segmentation runs locally with zero setup. Skipped (lean cloud build)
# when the bundle dir is absent. Build it first with build_localseg_env.py.
BUNDLE_DIR="${SPINE_HU_LOCALSEG_BUNDLE:-$ROOT/packaging/localseg-bundle}"
if [[ -d "$BUNDLE_DIR/localseg-env" ]]; then
  echo "==> Bundling local-seg runtime from $BUNDLE_DIR"
  python "$ROOT/packaging/bundle_localseg.py" --bundle "$BUNDLE_DIR" --app "$APP"
  # Adhoc-sign so the freshly-copied nested binaries launch on unsigned builds
  # (an existing signature over the .app is invalidated by adding files).
  echo "==> Adhoc-signing the .app (unsigned distribution)"
  codesign --force --deep --sign - "$APP" || \
    echo "WARN: adhoc codesign failed (continuing)"
else
  echo "==> No local-seg bundle at $BUNDLE_DIR; building LEAN (cloud) app"
fi

rm -f "$DMG"

if command -v create-dmg >/dev/null 2>&1; then
  echo "==> Packaging .dmg with create-dmg"
  create-dmg \
    --volname "$APP_NAME" \
    --volicon "$VOL_ICON" \
    --window-pos 200 120 \
    --window-size 640 400 \
    --icon-size 110 \
    --icon "$APP_NAME.app" 160 200 \
    --app-drop-link 480 200 \
    --no-internet-enable \
    "$DMG" "$APP"
else
  echo "==> create-dmg not found; using hdiutil fallback"
  STAGE="$(mktemp -d)"
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov \
    -format UDZO "$DMG"
  rm -rf "$STAGE"
fi

echo "==> Done: $DMG"
