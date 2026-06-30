"""Frozen-app entry point.

PyInstaller targets this script. It just launches the PySide6 GUI. Keeping the
entry point in `packaging/` (rather than calling the module directly) gives a
stable, import-safe target and a place to set frozen-only runtime tweaks.
"""
from __future__ import annotations

import os
import sys
import multiprocessing


def _bootstrap() -> None:
    # Needed so any child processes spawned by frozen builds re-exec correctly.
    multiprocessing.freeze_support()

    # Frozen apps have no console; make sure stdout/stderr never crash on write.
    if getattr(sys, "frozen", False):
        for stream_name in ("stdout", "stderr"):
            if getattr(sys, stream_name) is None:
                setattr(sys, stream_name, open(os.devnull, "w"))


def main() -> None:
    _bootstrap()
    from spine_hu_tool.app.viewer import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
