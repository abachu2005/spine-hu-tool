"""Deterministic body-habitus measurement from scout projections.

Produces, per vertebral level, the patient's left-right width (from the AP
projection) and anterior-posterior depth (from the lateral projection), in
millimeters, corrected for divergent-beam magnification.

Why the thresholds are all relative
-----------------------------------
A scout is a projection radiograph: its values are line integrals, not HU. Air
does not sit at -1000 -- on the project test data the background is near -450
and the body peaks around +700. So every threshold here is expressed relative
to a per-image background estimate (measured from the image border) and a
per-image body-attenuation reference (a high percentile). This keeps the
measurement stable across scanners, kVp, and patient size.

Why the CT couch needs no special model
---------------------------------------
In a lateral projection the beam runs edge-on along the length of the table, so
the couch shows up as a distinct low, flat plateau posterior to the patient
(about 8% of body attenuation on the test data). It therefore falls below the
body threshold on its own. Table rails can spike above it, so the body is taken
as the LARGEST contiguous run above threshold, which rejects them. In the AP
projection the beam crosses the couch perpendicular (a very short path) and it
is invisible, which is exactly what is observed.

Magnification
-------------
apparent = true * SOD / (SOD + d), where d is the offset of the body center
from the isocenter along the beam axis. Each view measures the offset the other
view needs -- the AP view measures the left-right center used by the lateral
view, and vice versa -- so the pair is solved by a short fixed-point iteration.
The correction is small (a few percent for a normally centered patient) and the
uncorrected value is always reported alongside it.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy import ndimage

from ..config import ScoutParams, SCOUT_SHOULDER_GIRDLE_LEVELS
from .loader import ScoutImage

SHOULDER_GIRDLE_LEVELS = frozenset(SCOUT_SHOULDER_GIRDLE_LEVELS)


def _background(pixels: np.ndarray, border_cols: int) -> float:
    """Air level, estimated from the untouched columns at both image edges."""
    n = max(1, min(int(border_cols), pixels.shape[1] // 4))
    edge = np.concatenate([pixels[:, :n].ravel(), pixels[:, -n:].ravel()])
    return float(np.median(edge))


def _odd(n: int) -> int:
    return n + 1 if n % 2 == 0 else n


def _row_reference(above: np.ndarray, row_spacing_mm: float,
                   params: ScoutParams) -> np.ndarray:
    """Per-row body-attenuation reference, smoothed along z.

    Referencing each row to its own anatomy is what makes the measurement
    comparable between scans with different coverage: a thoracic scout that
    includes the shoulders has a far higher peak attenuation than a lumbar one,
    and a single global reference would raise the threshold enough to shrink the
    same patient by ~10 mm.
    """
    ref = np.percentile(above, params.ref_percentile, axis=1)
    k = _odd(int(round(params.ref_smooth_mm / max(row_spacing_mm, 1e-6))))
    if k >= 3:
        ref = ndimage.median_filter(ref, size=k, mode="nearest")
    return ref


def _largest_run(mask: np.ndarray):
    """(first, last) index of the longest contiguous True run, or None."""
    if not mask.any():
        return None
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    idx = np.flatnonzero(lab == int(sizes.argmax()))
    return int(idx[0]), int(idx[-1])


def body_profile(scout: ScoutImage,
                 params: Optional[ScoutParams] = None) -> dict:
    """Per-row body extent for one projection, in patient millimeters.

    Returns arrays indexed by image row: `z_mm`, `lo_mm`/`hi_mm` (the two body
    boundaries along the measured axis), `width_mm`, `center_mm`, and a boolean
    `clipped` marking rows whose extent reaches the image edge.
    """
    params = params or ScoutParams()
    px = scout.pixels
    bg = _background(px, params.bg_border_cols)
    above = px - bg
    ref = _row_reference(above, scout.row_spacing_mm, params)
    thr = params.body_threshold_frac * ref

    n_rows, n_cols = px.shape
    lo = np.full(n_rows, np.nan)
    hi = np.full(n_rows, np.nan)
    clipped = np.zeros(n_rows, dtype=bool)
    edge_cols = max(1, int(round(params.edge_margin_mm / scout.col_spacing_mm)))

    for r in range(n_rows):
        run = _largest_run(above[r, :] > thr[r])
        if run is None:
            continue
        c0, c1 = run
        p0 = float(scout.column_to_patient(c0))
        p1 = float(scout.column_to_patient(c1))
        lo[r], hi[r] = min(p0, p1), max(p0, p1)
        clipped[r] = (c0 <= edge_cols) or (c1 >= n_cols - 1 - edge_cols)

    width = hi - lo
    # Light median smoothing along z suppresses per-row detection jitter without
    # moving real anatomy (the smoothing length is a few millimeters).
    k = _odd(int(round(params.smooth_mm / max(scout.row_spacing_mm, 1e-6))))
    if k >= 3:
        valid = ~np.isnan(width)
        if valid.any():
            filled = np.where(valid, width, np.nanmedian(width[valid]))
            sm = ndimage.median_filter(filled, size=k, mode="nearest")
            width = np.where(valid, sm, np.nan)

    return {
        "z_mm": scout.row_to_z(np.arange(n_rows)),
        "lo_mm": lo, "hi_mm": hi, "width_mm": width,
        "center_mm": (lo + hi) / 2.0,
        "clipped": clipped,
        "background": bg,
        "reference": float(np.median(ref)),
        "threshold": float(np.median(thr)),
        "view": scout.view, "axis": scout.axis, "beam_axis": scout.beam_axis,
    }


def _band_average(prof: dict, z0: float, z1: float) -> dict:
    """Average a profile over the z band [z0, z1] (nearest row if too thin)."""
    z = prof["z_mm"]
    sel = (z >= min(z0, z1)) & (z <= max(z0, z1))
    if not sel.any():
        k = int(np.argmin(np.abs(z - (z0 + z1) / 2.0)))
        sel = np.zeros_like(z, dtype=bool)
        sel[k] = True
    w = prof["width_mm"][sel]
    c = prof["center_mm"][sel]
    ok = ~np.isnan(w)
    if not ok.any():
        return {"width_mm": float("nan"), "center_mm": float("nan"),
                "clipped": bool(prof["clipped"][sel].any()), "n_rows": 0,
                "spread_mm": float("nan")}
    return {
        "width_mm": float(np.median(w[ok])),
        "center_mm": float(np.median(c[ok])),
        "clipped": bool(prof["clipped"][sel].any()),
        "n_rows": int(ok.sum()),
        # Rows within a level are millimetres apart, so a large spread means the
        # outline jumped objects rather than following anatomy.
        "spread_mm": float(w[ok].max() - w[ok].min()),
    }


def _magnification(profiles: dict, bands: dict, params: ScoutParams) -> dict:
    """Solve the mutual AP/lateral magnification correction at one level.

    Each view's scale factor depends on the body-center offset along its beam
    axis, which is what the *other* view measures; a couple of iterations is
    ample because the coupling is weak.
    """
    factors = {v: 1.0 for v in bands}
    sod = None
    for v, prof in profiles.items():
        s = prof.get("sod_mm")
        if s:
            sod = s
    if not sod:
        return factors                      # no geometry -> report uncorrected

    # Which view supplies the offset each other view needs: a view's beam axis
    # is the patient axis the other view measures along.
    supplier = {}
    for v, prof in profiles.items():
        for other, oprof in profiles.items():
            if other != v and oprof["axis"] == prof["beam_axis"]:
                supplier[v] = other
    for _ in range(max(1, int(params.mag_iters))):
        new = {}
        for v in bands:
            src = supplier.get(v)
            if src is None or np.isnan(bands[src]["center_mm"]):
                new[v] = factors.get(v, 1.0)
                continue
            d = bands[src]["center_mm"] * factors.get(src, 1.0)
            new[v] = float((sod + params.beam_sign * d) / sod)
        factors = new
    return factors


def scout_json(block: Optional[dict]) -> Optional[dict]:
    """The JSON-serializable part of a scout block (drops cached arrays)."""
    if not block:
        return block
    return {k: v for k, v in block.items() if not k.startswith("_")}


def level_z_ranges(seg: np.ndarray, volume, levels=None) -> dict:
    """Superior-inferior extent (z_min, z_max in patient mm) of each level.

    Uses the volume origin/spacing, so it is only meaningful for a volume loaded
    from DICOM (where the origin is real); a NIfTI-loaded volume with a default
    origin would place the levels in the wrong frame.
    """
    from ..segmentation.totalseg_runner import label_id_for, vertebra_labels
    if levels is None:
        levels = [lvl for _id, lvl in vertebra_labels(seg)]
    z0 = float(volume.origin[2])
    sz = float(volume.spacing[2])
    out = {}
    for lvl in levels:
        try:
            lab = label_id_for(lvl)
        except KeyError:
            continue
        zs = np.flatnonzero((seg == lab).any(axis=(0, 1)))
        if zs.size == 0:
            continue
        out[lvl] = (z0 + float(zs.min()) * sz, z0 + float(zs.max()) * sz)
    return out


def measure_habitus(scouts: list[ScoutImage], levels: dict,
                    params: Optional[ScoutParams] = None) -> dict:
    """Per-level body width/depth from the available scout projections.

    `levels` maps level name -> (z_min_mm, z_max_mm). Returns a dict with the
    per-view profiles' summaries and a `levels` block carrying, per level,
    `body_width_lr_mm`, `body_depth_ap_mm`, `body_effective_diameter_mm`, the
    applied magnification factors, and any QC warnings.
    """
    params = params or ScoutParams()
    if not scouts:
        return {"available": False, "views": {}, "levels": {},
                "warnings": ["no scout (localizer) series in this study"],
                "params": params.to_dict()}

    profiles = {}
    for s in scouts:
        prof = body_profile(s, params)
        prof["sod_mm"] = s.sod_mm
        prof["sid_mm"] = s.sid_mm
        profiles[s.view] = prof

    global_warnings = []
    if "AP" not in profiles:
        global_warnings.append("no AP scout: left-right width unavailable")
    if "LAT" not in profiles:
        global_warnings.append("no lateral scout: anterior-posterior depth unavailable")

    out_levels = {}
    for lvl, (za, zb) in levels.items():
        # Average over the central band of the level so endplate-adjacent rows
        # (where the neighbouring vertebra starts) do not skew the value.
        mid = (za + zb) / 2.0
        half = abs(zb - za) / 2.0 * params.level_band_frac
        z0, z1 = mid - half, mid + half

        bands = {v: _band_average(p, z0, z1) for v, p in profiles.items()}
        factors = _magnification(profiles, bands, params)

        entry, warns = {}, []
        for view, key in (("AP", "body_width_lr_mm"), ("LAT", "body_depth_ap_mm")):
            b = bands.get(view)
            if b is None or np.isnan(b["width_mm"]):
                continue
            f = factors.get(view, 1.0)
            raw = b["width_mm"]
            val = raw * f
            entry[key] = round(val, 1)
            entry[key.replace("_mm", "_uncorrected_mm")] = round(raw, 1)
            entry[f"scout_{view.lower()}_magnification"] = round(f, 4)
            entry[f"scout_{view.lower()}_center_mm"] = round(b["center_mm"] * f, 1)
            if b["clipped"]:
                warns.append(f"scout {view} projection reaches the image edge; "
                             "body outline may be cut off")
            if b["spread_mm"] > params.max_row_spread_mm:
                warns.append(f"scout {view} outline varies by "
                             f"{b['spread_mm']:.0f} mm across this level; "
                             "check the exported scout overlay")
            if not (params.min_plausible_mm <= val <= params.max_plausible_mm):
                warns.append(f"scout {view} extent {val:.0f} mm is outside the "
                             "plausible torso range (arms/shoulders in field?)")
            if abs(b["center_mm"] * f) > params.max_center_offset_mm:
                warns.append(f"scout {view} body center is "
                             f"{b['center_mm'] * f:.0f} mm off isocenter; "
                             "magnification correction is less reliable")
        if lvl in SHOULDER_GIRDLE_LEVELS and "body_width_lr_mm" in entry:
            warns.append("the shoulder girdle and upper arms project over the "
                         "torso at this level, so the left-right width is a "
                         "shoulder width, not a torso width")
        w = entry.get("body_width_lr_mm")
        d = entry.get("body_depth_ap_mm")
        if w and d and w > 0 and d > 0:
            # Effective diameter (AAPM 220): the diameter of the circle with the
            # same cross-sectional area as the elliptical torso.
            entry["body_effective_diameter_mm"] = round(float(np.sqrt(w * d)), 1)
        if entry:
            entry["scout_warnings"] = warns
            entry["scout_band_z_mm"] = [round(z0, 1), round(z1, 1)]
            out_levels[lvl] = entry

    return {
        "available": True,
        # Kept for overlay rendering only; stripped before serialization.
        "_scouts": scouts,
        "_profiles": profiles,
        "views": {v: s.summary() for v, s in ((s.view, s) for s in scouts)},
        "profiles": {v: {"background": round(p["background"], 1),
                         "reference": round(p["reference"], 1),
                         "threshold": round(p["threshold"], 1)}
                     for v, p in profiles.items()},
        "levels": out_levels,
        "warnings": global_warnings,
        "params": params.to_dict(),
    }


def habitus_for_case(folder: str, volume, seg, results=None,
                     params: Optional[ScoutParams] = None,
                     candidates: Optional[list] = None) -> dict:
    """End-to-end: find this study's scouts and measure every level present."""
    from .loader import find_scouts
    study_uid = (volume.metadata or {}).get("study_uid") or None
    scouts = find_scouts(folder, study_uid=study_uid, candidates=candidates)
    levels = level_z_ranges(seg, volume,
                            list(results) if results is not None else None)
    return measure_habitus(scouts, levels, params)
