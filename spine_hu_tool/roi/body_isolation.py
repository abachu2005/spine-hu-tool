"""Vertebral-body isolation: strip posterior elements from a full vertebra mask.

Deterministic morphological approach (no ML): a morphological opening removes
the thin posterior connections (pedicles), disconnecting the posterior elements
(lamina, spinous/transverse processes) from the body. The most anterior large
component is kept and its volume recovered within the original mask.

Returns (body_mask, flags) where flags records QC concerns for downstream
review rather than failing silently.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, label

from ..geometry.coords import ball, largest_cc


def _most_anterior_component(mask: np.ndarray, min_vox: int = 50) -> np.ndarray:
    """Keep the connected component with the smallest mean y (most anterior)."""
    lab, n = label(mask)
    if n == 0:
        return mask
    best, best_y = None, np.inf
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_vox:
            continue
        ymean = np.argwhere(comp)[:, 1].mean()
        if ymean < best_y:
            best_y, best = ymean, comp
    return best if best is not None else largest_cc(mask)


# Candidate opening radii (mm), largest first. A single fixed size is unreliable
# across vertebra sizes/resolutions -- too big annihilates small vertebrae, too
# small leaves posterior elements attached -- so we search and pick adaptively.
_OPEN_CANDIDATES_MM = (6.0, 5.0, 4.0, 3.0, 2.5)
_RATIO_TARGET = 0.55             # a clean body keeps roughly half the vertebra
_RATIO_BAND = (0.35, 0.80)       # acceptable body/full ratio


def isolate_body(full_mask: np.ndarray, spacing, open_mm: float | None = None) -> tuple[np.ndarray, dict]:
    """Strip posterior elements via an *adaptive* morphological opening.

    Tries a range of opening radii and keeps the candidate whose body/full
    ratio best lands in a sane band (posterior elements removed, body intact).
    Never falls back to the full vertebra (which would drag the centroid into
    the canal); if nothing isolates, flags it so the level can be excluded.
    """
    M = full_mask.astype(bool)
    flags: dict = {"warnings": []}
    nvox = int(M.sum())
    if nvox < 100:
        flags["warnings"].append("vertebra mask too small to isolate")
        flags["too_small"] = True
        return M, flags

    candidates = list(_OPEN_CANDIDATES_MM)
    if open_mm is not None and open_mm not in candidates:
        candidates = sorted(set(candidates) | {float(open_mm)}, reverse=True)

    results = []
    for om in candidates:
        b = ball(om, spacing)
        eroded = binary_erosion(M, b)
        if not eroded.any():
            continue                       # this opening annihilates the mask
        core = _most_anterior_component(eroded)
        body = binary_fill_holes(binary_dilation(core, b) & M)
        body = largest_cc(body)
        ratio = body.sum() / nvox
        results.append((om, ratio, body))

    if not results:
        flags["warnings"].append("erosion removed entire mask; body not isolable")
        flags["not_isolable"] = True
        return largest_cc(M), flags

    in_band = [r for r in results if _RATIO_BAND[0] <= r[1] <= _RATIO_BAND[1]]
    pool = in_band or results
    om, ratio, body = min(pool, key=lambda r: abs(r[1] - _RATIO_TARGET))

    flags["body_full_ratio"] = float(ratio)
    flags["open_mm"] = float(om)
    if ratio < 0.30:
        flags["warnings"].append(f"body/full ratio very low ({ratio:.2f}); isolation suspect")
        flags["isolation_suspect"] = True
    elif ratio > 0.90:
        flags["warnings"].append(f"body/full ratio high ({ratio:.2f}); posterior elements may remain")

    # "partial" is judged on the BODY, not the full vertebra: thoracic transverse
    # processes legitimately reach a tight FOV edge while the body is fully
    # captured. Only a clipped body invalidates the measurement.
    if _touches_edge(body):
        flags["warnings"].append("vertebral body touches scan field edge (partial)")
        flags["partial"] = True

    return body, flags


def _touches_edge(mask: np.ndarray) -> bool:
    for ax in range(mask.ndim):
        first = np.take(mask, 0, axis=ax)
        last = np.take(mask, mask.shape[ax] - 1, axis=ax)
        if first.any() or last.any():
            return True
    return False
