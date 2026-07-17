"""Lightweight, dependency-free system probes for local-run safety checks.

Used to decide whether a machine can realistically run full-resolution local
segmentation (which needs a lot of RAM) or should be steered to fast (3 mm)
mode before the user starts a run that would swap/crash.
"""
from __future__ import annotations
import os
import sys
from typing import Optional

# Full-res TotalSegmentator (1.5 mm ensemble) needs a lot of RAM; below this we
# warn the user that a full-res run is likely to swap or crash and offer fast.
FULLRES_MIN_RAM_GB = 12.0


def total_ram_gb() -> Optional[float]:
    """Total physical RAM in GB, or None if it can't be determined.

    Uses ``psutil`` when available, otherwise per-OS fallbacks so it works in a
    frozen app with no extra dependency:
      - Linux: ``sysconf`` pages * page size
      - macOS: ``sysctl hw.memsize`` via ``os.sysconf`` when present, else ctypes
      - Windows: ``GlobalMemoryStatusEx`` via ctypes
    """
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    try:
        if sys.platform.startswith("win"):
            return _windows_ram_gb()
        # Linux and most Unix expose these sysconf names.
        if hasattr(os, "sysconf"):
            names = os.sysconf_names  # type: ignore[attr-defined]
            if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
                page = os.sysconf("SC_PAGE_SIZE")
                pages = os.sysconf("SC_PHYS_PAGES")
                if page > 0 and pages > 0:
                    return (page * pages) / (1024 ** 3)
        if sys.platform == "darwin":
            return _macos_ram_gb()
    except Exception:
        return None
    return None


def _windows_ram_gb() -> Optional[float]:
    import ctypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return stat.ullTotalPhys / (1024 ** 3)
    return None


def _macos_ram_gb() -> Optional[float]:
    import subprocess
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return None


def fullres_ram_warning(min_gb: float = FULLRES_MIN_RAM_GB) -> Optional[str]:
    """Return a warning message if this machine likely can't run full-res, else None.

    Conservative: if RAM can't be determined, returns None (no scary warning on
    a machine we can't measure).
    """
    ram = total_ram_gb()
    if ram is None:
        return None
    if ram < min_gb:
        return (
            f"This computer has about {ram:.0f} GB of RAM. Full-resolution local "
            f"segmentation typically needs {min_gb:.0f} GB or more and may run "
            "out of memory and crash on this machine. Use Fast (3 mm) mode "
            "instead \u2014 it is much lighter and the measurement ROI placement "
            "is essentially unaffected.")
    return None
