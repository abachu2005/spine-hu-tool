"""Synthetic phantoms for deterministic, ML-free unit tests.

Convention matches the package: arrays [x, y, z], y increasing = posterior,
z increasing = superior. A vertebra phantom = anterior body ellipsoid (with a
denser cortical shell) + a posterior-element blob connected by a thin bridge.
"""
from __future__ import annotations
import numpy as np


def _phys_grid(shape, spacing):
    xs = np.arange(shape[0]) * spacing[0]
    ys = np.arange(shape[1]) * spacing[1]
    zs = np.arange(shape[2]) * spacing[2]
    return np.meshgrid(xs, ys, zs, indexing="ij")


def _ellipsoid(X, Y, Z, center, radii):
    cx, cy, cz = center
    rx, ry, rz = radii
    return (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2) <= 1.0


def make_vertebra_phantom(spacing=(1.0, 1.0, 1.0),
                          extent_mm=(60, 70, 50),
                          trab_hu=150.0, cortex_hu=700.0, post_hu=450.0):
    """Return (full_mask, body_mask_true, hu) for a synthetic vertebra."""
    shape = tuple(int(round(e / s)) for e, s in zip(extent_mm, spacing))
    X, Y, Z = _phys_grid(shape, spacing)

    body_c = (30.0, 22.0, 25.0)
    body_r = (16.0, 12.0, 13.0)
    body = _ellipsoid(X, Y, Z, body_c, body_r)
    inner = _ellipsoid(X, Y, Z, body_c, tuple(r - 2.0 for r in body_r))
    cortex = body & ~inner

    post = _ellipsoid(X, Y, Z, (30.0, 50.0, 25.0), (9.0, 8.0, 10.0))
    bridge = _ellipsoid(X, Y, Z, (30.0, 37.0, 25.0), (3.0, 8.0, 3.0))

    full = body | post | bridge
    hu = np.full(shape, -200.0, dtype=np.float32)   # background "soft tissue"
    hu[inner] = trab_hu
    hu[cortex] = cortex_hu
    hu[post | bridge] = post_hu
    return full.astype(bool), body.astype(bool), hu


def make_scout_pair(width_lr_mm=360.0, depth_ap_mm=280.0,
                    center=(0.0, 20.0), z_range=(-200.0, 200.0),
                    col_spacing=0.6, row_spacing=0.55,
                    sid_mm=950.0, sod_mm=540.0, background=-450.0,
                    body_peak=900.0, table=True, magnify=True):
    """An AP + lateral scout pair through an elliptical torso of known size.

    The torso is a z-invariant ellipse centered at ``center`` = (x, y) in patient
    millimeters, projected the way a real scanner does: each ray's value is
    proportional to its chord length through the ellipse, and (when `magnify`)
    the projection is scaled by the divergent-beam factor SOD / (SOD + d) for
    the body-center offset d along that view's beam axis. `table` adds the flat
    low-attenuation couch slab that only the lateral view sees.

    Returns (ap_scout, lat_scout) as :class:`ScoutImage` objects, so the whole
    measurement stack can be exercised with no DICOM and a known right answer.
    """
    from ..scout.loader import ScoutImage

    cx, cy = center
    half = {0: width_lr_mm / 2.0, 1: depth_ap_mm / 2.0}
    ctr = {0: cx, 1: cy}
    n_cols, n_rows = 888, int(round((z_range[1] - z_range[0]) / row_spacing))
    span = n_cols * col_spacing

    out = []
    for axis in (0, 1):
        beam_axis = 1 - axis
        # An object offset toward the source (negative d with this sign
        # convention) casts a larger shadow: apparent = true * SOD / (SOD + d).
        mag = 1.0
        if magnify and sod_mm:
            mag = sod_mm / (sod_mm + ctr[beam_axis])
        # Columns run along +axis for the AP view and -y for the lateral view,
        # matching the ImageOrientationPatient of a real CT localizer.
        sign = 1.0 if axis == 0 else -1.0
        origin = [0.0, 0.0, z_range[1]]
        origin[axis] = -sign * span / 2.0
        coord = origin[axis] + sign * np.arange(n_cols) * col_spacing

        # Apparent (projected) ellipse half-width and center at the image plane.
        a = half[axis] * mag
        c = ctr[axis] * mag
        u = (coord - c) / a
        chord = np.sqrt(np.clip(1.0 - u ** 2, 0.0, None))
        row = background + body_peak * chord
        pixels = np.tile(row.astype(np.float32), (n_rows, 1))

        if table and axis == 1:
            # Couch: a flat slab posterior to the patient, at ~8% of body peak,
            # exactly the signature seen on the real scanners.
            slab = (coord > ctr[1] + half[1] + 8.0) & (coord < ctr[1] + half[1] + 55.0)
            pixels[:, slab] = background + 0.08 * body_peak

        row_dir = [0.0, 0.0, 0.0]
        row_dir[axis] = sign
        out.append(ScoutImage(
            pixels=pixels, view="AP" if axis == 0 else "LAT",
            axis=axis, beam_axis=beam_axis,
            row_spacing_mm=row_spacing, col_spacing_mm=col_spacing,
            origin=tuple(origin), row_dir=tuple(row_dir),
            col_dir=(0.0, 0.0, -1.0), sid_mm=sid_mm, sod_mm=sod_mm,
            description="synthetic"))
    return out[0], out[1]


def make_tilted_ellipsoid(spacing=(1.0, 1.0, 1.0), shape=(50, 50, 50),
                          radii=(8.0, 8.0, 18.0), tilt_deg=20.0):
    """Tall ellipsoid rotated by `tilt_deg` about the x-axis (y-z plane)."""
    X, Y, Z = _phys_grid(shape, spacing)
    c = (np.array(shape) * np.array(spacing)) / 2.0
    th = np.deg2rad(tilt_deg)
    yy = (Y - c[1]) * np.cos(th) + (Z - c[2]) * np.sin(th)
    zz = -(Y - c[1]) * np.sin(th) + (Z - c[2]) * np.cos(th)
    xx = X - c[0]
    mask = (xx / radii[0]) ** 2 + (yy / radii[1]) ** 2 + (zz / radii[2]) ** 2 <= 1.0
    # long axis (where xx=yy=0): dY/dZ = -tan(th) -> direction (0, -sin, cos)
    true_si = np.array([0.0, -np.sin(th), np.cos(th)])
    return mask.astype(bool), true_si
