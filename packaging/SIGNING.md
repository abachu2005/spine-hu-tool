# Code signing & notarization (optional)

The installers work **unsigned**, but unsigned apps trip OS gatekeepers. This
doc covers the optional signing setup and the unsigned fallbacks physicians can
use today.

## Why sign?

| OS | Unsigned behavior | Fixed by |
| --- | --- | --- |
| macOS | "App is damaged / from an unidentified developer" Gatekeeper block | Developer ID signing + notarization |
| Windows | SmartScreen "Windows protected your PC" warning | Authenticode code-signing cert |
| Linux | none (AppImage runs as-is) | n/a |

## Unsigned fallbacks (no cert needed)

- **macOS**: right-click the app → **Open** → **Open** (once). Or:
  `xattr -dr com.apple.quarantine "/Applications/Spine HU Tool.app"`
- **Windows**: on the SmartScreen dialog click **More info → Run anyway**.
- **Linux**: `chmod +x SpineHUTool-linux-x86_64.AppImage` then run it.

## macOS — Developer ID signing + notarization

Requires an Apple Developer account ($99/yr) and a **Developer ID Application**
certificate.

```bash
# 1. sign the app (hardened runtime is required for notarization)
codesign --deep --force --options runtime --timestamp \
  --sign "Developer ID Application: Your Name (TEAMID)" \
  "packaging/dist/Spine HU Tool.app"

# 2. build the dmg (packaging/macos/build_dmg.sh) and notarize it
xcrun notarytool submit "packaging/dist/Spine HU Tool.dmg" \
  --apple-id "you@example.com" --team-id "TEAMID" \
  --password "app-specific-password" --wait

# 3. staple the ticket so it verifies offline
xcrun stapler staple "packaging/dist/Spine HU Tool.dmg"
```

In CI, store as repo secrets and decode the cert into the keychain:
`MACOS_CERT_P12` (base64), `MACOS_CERT_PASSWORD`, `MACOS_SIGN_IDENTITY`,
`APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD`. The release workflow runs the
signing step only when these secrets are present; otherwise it ships unsigned.

## Windows — Authenticode signing

Requires a code-signing certificate (OV or, to avoid SmartScreen reputation
warnings, EV).

```powershell
& "C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe" sign `
  /f cert.pfx /p $env:CERT_PASSWORD /tr http://timestamp.digicert.com `
  /td sha256 /fd sha256 "packaging\dist\Spine HU Tool\Spine HU Tool.exe"
```

Sign **before** running `iscc` so the bundled exe is signed, then sign the
produced `SpineHUTool-Setup.exe` as well. In CI provide `WINDOWS_CERT_PFX`
(base64) and `WINDOWS_CERT_PASSWORD` secrets.

## Notes

- Signing is intentionally **off by default** so anyone can build installers
  without certificates.
- The first-run dialog in the app reminds users this is pilot / non-PHI
  software regardless of signing status.
