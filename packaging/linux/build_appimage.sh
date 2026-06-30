#!/usr/bin/env bash
# Build a Linux AppImage from the PyInstaller onedir bundle.
#
# Prereqs:
#   pip install pyinstaller
#   appimagetool on PATH (auto-downloaded below if missing)
#
# Usage (from repo root):
#   bash packaging/linux/build_appimage.sh
set -euo pipefail

APP_NAME="Spine HU Tool"
APP_ID="spine-hu-tool"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST="$ROOT/packaging/dist"
BUNDLE="$DIST/$APP_NAME"
APPDIR="$DIST/SpineHU.AppDir"
OUT="$DIST/SpineHUTool-x86_64.AppImage"

# 1. build the onedir bundle if missing
if [[ ! -d "$BUNDLE" ]]; then
  echo "==> Building bundle with PyInstaller"
  pyinstaller "$ROOT/packaging/spine_hu.spec" --noconfirm \
    --distpath "$DIST" --workpath "$ROOT/packaging/build"
fi

# 2. assemble the AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -R "$BUNDLE/." "$APPDIR/usr/bin/"

# icon (top-level + hicolor)
cp "$ROOT/packaging/icons/spine_hu.png" "$APPDIR/$APP_ID.png"
cp "$ROOT/packaging/icons/spine_hu.png" \
   "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"

# .desktop entry
cat > "$APPDIR/$APP_ID.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Exec=AppRun
Icon=$APP_ID
Categories=Science;MedicalSoftware;
Comment=Reproducible trabecular vertebral-HU measurement from CT
Terminal=false
EOF
cp "$APPDIR/$APP_ID.desktop" "$APPDIR/usr/share/applications/"

# AppRun launcher
cat > "$APPDIR/AppRun" <<EOF
#!/usr/bin/env bash
HERE="\$(dirname "\$(readlink -f "\${0}")")"
exec "\$HERE/usr/bin/$APP_NAME" "\$@"
EOF
chmod +x "$APPDIR/AppRun"

# 3. get appimagetool
TOOL="$(command -v appimagetool || true)"
if [[ -z "$TOOL" ]]; then
  echo "==> Downloading appimagetool"
  TOOL="$DIST/appimagetool-x86_64.AppImage"
  curl -L -o "$TOOL" \
    "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$TOOL"
fi

# 4. build the AppImage (ARCH required by appimagetool)
echo "==> Building AppImage"
ARCH=x86_64 "$TOOL" "$APPDIR" "$OUT"

echo "==> Done: $OUT"
