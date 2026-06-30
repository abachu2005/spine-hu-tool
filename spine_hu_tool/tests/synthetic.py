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
