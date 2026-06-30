#!/usr/bin/env bash
# Publish the download page + whatever installers exist locally to the public
# GCS bucket. Lets you push the macOS .dmg you can build today, before CI is
# wired up. CI does the same thing automatically on tag push.
#
# Usage (from repo root):
#   bash packaging/upload_to_gcs.sh                 # uses default bucket
#   GCS_BUCKET=my-bucket bash packaging/upload_to_gcs.sh
set -euo pipefail

BUCKET="${GCS_BUCKET:-spine-hu-tool-downloads}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/packaging/dist"
DL="$ROOT/packaging/download"

echo "==> Publishing to gs://$BUCKET"

# 1. download page assets (short cache so updates show up immediately)
gcloud storage cp "$DL/index.html"   "gs://$BUCKET/index.html" \
  --cache-control="no-cache"
gcloud storage cp "$DL/version.json" "gs://$BUCKET/version.json" \
  --cache-control="no-cache"
gcloud storage cp "$ROOT/packaging/icons/spine_hu.png" "gs://$BUCKET/icon.png" \
  --cache-control="public, max-age=86400" || true

# 2. installers (long cache; immutable per release)
upload() {  # <local> <remote-name>
  local src="$1" name="$2"
  if [[ -f "$src" ]]; then
    echo "   - $name"
    gcloud storage cp "$src" "gs://$BUCKET/latest/$name" \
      --cache-control="public, max-age=3600"
  fi
}

upload "$DIST/Spine HU Tool.dmg"               "SpineHUTool-macos.dmg"
upload "$DIST/SpineHUTool-Setup.exe"           "SpineHUTool-windows-setup.exe"
upload "$DIST/SpineHUTool-x86_64.AppImage"     "SpineHUTool-linux-x86_64.AppImage"

echo "==> Done."
echo "    Download page: https://storage.googleapis.com/$BUCKET/index.html"
