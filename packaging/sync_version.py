"""Propagate the one true version number to the files that can't compute it.

`spine_hu_tool.__version__` is the source of truth: it is what every export
stamps into `reproducibility.json` as `tool_version`, so it is the number that
has to be right for a measurement to be traceable to the code that produced it.

`pyproject.toml` and `spine_hu.spec` read that attribute directly. Two consumers
cannot -- Inno Setup has no way to import Python, and the download page's
`version.json` is served as a static file -- so they carry a copy, and this
script keeps the copies honest:

    python packaging/sync_version.py            # rewrite the copies
    python packaging/sync_version.py --check    # fail if any copy has drifted
    python packaging/sync_version.py --set 0.1.4    # bump everything at once

The --check mode runs in CI before the installers are built, because a build
that advertises one version while stamping another into its output is worse
than a build that fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INIT = os.path.join(ROOT, "spine_hu_tool", "__init__.py")
ISS = os.path.join(ROOT, "packaging", "windows", "installer.iss")
VERSION_JSON = os.path.join(ROOT, "packaging", "download", "version.json")

_INIT_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)
_ISS_RE = re.compile(r'^(#define\s+AppVersion\s+")([^"]*)(")', re.M)


def read_version(root: str = ROOT) -> str:
    """The version declared in spine_hu_tool/__init__.py.

    Parsed rather than imported so the build can read it without importing the
    package (and its dependencies) first.
    """
    path = os.path.join(root, "spine_hu_tool", "__init__.py")
    with open(path, encoding="utf-8") as fh:
        m = _INIT_RE.search(fh.read())
    if not m:
        raise RuntimeError(f"no __version__ found in {path}")
    return m.group(1)


def _iss_version(text: str) -> str | None:
    m = _ISS_RE.search(text)
    return m.group(2) if m else None


def _set_source(version: str) -> None:
    with open(INIT, encoding="utf-8") as fh:
        text = fh.read()
    new, n = _INIT_RE.subn(f'__version__ = "{version}"', text)
    if n != 1:
        raise RuntimeError(f"expected exactly one __version__ in {INIT}, found {n}")
    with open(INIT, "w", encoding="utf-8") as fh:
        fh.write(new)


def _sync_iss(version: str, write: bool) -> str | None:
    """Returns a description of the drift, or None when already in sync."""
    with open(ISS, encoding="utf-8") as fh:
        text = fh.read()
    current = _iss_version(text)
    if current is None:
        raise RuntimeError(f"no '#define AppVersion' in {ISS}")
    if current == version:
        return None
    if write:
        with open(ISS, "w", encoding="utf-8") as fh:
            fh.write(_ISS_RE.sub(rf"\g<1>{version}\g<3>", text))
    return f"packaging/windows/installer.iss: AppVersion {current!r}"


def _sync_version_json(version: str, write: bool) -> str | None:
    with open(VERSION_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    current = data.get("version")
    if current == version:
        return None
    if write:
        data["version"] = version
        with open(VERSION_JSON, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
    return f"packaging/download/version.json: version {current!r}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; change nothing")
    ap.add_argument("--set", dest="new_version", metavar="X.Y.Z",
                    help="set the source version, then propagate it")
    ap.add_argument("--expect", metavar="X.Y.Z",
                    help="also require the source version to equal this "
                         "(CI passes the release tag, so a tag can't ship code "
                         "that identifies itself as a different version)")
    args = ap.parse_args(argv)

    if args.new_version:
        if args.check:
            ap.error("--set and --check are mutually exclusive")
        _set_source(args.new_version)

    version = read_version()
    if args.expect and args.expect != version:
        print(f"expected version {args.expect!r} but spine_hu_tool.__version__ "
              f"is {version!r}.\nrun: python packaging/sync_version.py --set "
              f"{args.expect}", file=sys.stderr)
        return 1

    write = not args.check
    drift = [d for d in (_sync_iss(version, write),
                         _sync_version_json(version, write)) if d]

    if args.check:
        if drift:
            print(f"version drift from spine_hu_tool.__version__ == {version!r}:",
                  file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            print("\nrun: python packaging/sync_version.py", file=sys.stderr)
            return 1
        print(f"version {version} is consistent across all files.")
        return 0

    if drift:
        for d in drift:
            print(f"updated {d.split(':')[0]} -> {version}")
    else:
        print(f"version {version} already consistent; nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
