"""Per-level metal/streak tagging and clean-level selection.

A level with metal voxels inside it is excluded. Levels within the streak
buffer, or immediately adjacent to an instrumented level, are downgraded to
"review" (streak/bloom corrupts neighbors even without metal inside them).
"""
from __future__ import annotations
import numpy as np

from .. import config
from ..geometry.coords import within_mm
from ..segmentation.totalseg_runner import vertebra_labels, VERT_ORDER


def tag_levels(hu: np.ndarray, seg: np.ndarray, spacing,
               near_pct_threshold: float = 2.0) -> list[dict]:
    metal = hu > config.METAL_HU_THRESHOLD
    buf = within_mm(metal, config.STREAK_BUFFER_MM, spacing)

    present = vertebra_labels(seg)
    rows: list[dict] = []
    for lab, short in present:
        m = seg == lab
        n = int(m.sum())
        if n == 0:
            continue
        metal_in = int((m & metal).sum())
        near = float((m & buf).sum()) / n * 100.0 if n else 0.0
        if metal_in > 0:
            status = "excluded"
        elif near > near_pct_threshold:
            status = "review"
        else:
            status = "clean"
        rows.append({"level": short, "label_id": lab, "voxels": n,
                     "metal_in": metal_in, "near_metal_pct": round(near, 1),
                     "status": status})

    _buffer_adjacent(rows)
    return rows


def _buffer_adjacent(rows: list[dict]) -> None:
    """Bump clean neighbors of excluded levels to 'review' (adjacent contamination)."""
    by_level = {r["level"]: r for r in rows}
    excluded = [r["level"] for r in rows if r["status"] == "excluded"]
    for lev in excluded:
        if lev not in VERT_ORDER:
            continue
        i = VERT_ORDER.index(lev)
        for j in (i - 1, i + 1):
            if 0 <= j < len(VERT_ORDER):
                nb = VERT_ORDER[j]
                if nb in by_level and by_level[nb]["status"] == "clean":
                    by_level[nb]["status"] = "review"
                    by_level[nb].setdefault("notes", []).append(
                        f"adjacent to instrumented {lev}")


def clean_levels(rows: list[dict]) -> list[str]:
    return [r["level"] for r in rows if r["status"] == "clean"]
