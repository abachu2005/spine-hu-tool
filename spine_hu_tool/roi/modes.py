"""ROI placement modes.

Default / reported mode: centroid_volume_sphere -- a volume-proportional 3D
sphere anchored at the vertebral-body centroid, reported as the median HU.
Placement is HU-independent (reproducible by design); the radius scales with
body volume (Westerhoff's size-scaling). The lowest_attenuation_sphere
(Westerhoff's screening ROI) and axial_ellipse_2d (Jang 2D) are kept as
reproducibility / sensitivity comparators.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

from ..config import ROIParams
from ..geometry.coords import make_sphere, ball
from ..geometry.local_axes import compute_local_axes
from .distance import (body_distance, make_inner, central_height_mask,
                       anterior_mask, proportional_margin)


def _clamp_center_to_body(center, body, dist):
    """Ensure the center voxel is inside the body; if not, use the deepest point."""
    c = tuple(int(round(v)) for v in center)
    inside = all(0 <= c[i] < body.shape[i] for i in range(3)) and body[c]
    if inside:
        return c
    return tuple(int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))


def centroid_sphere(body, spacing, params: ROIParams, hu=None):
    """Trabecular ROI at the body centroid: a sphere whose radius is set by the
    *in-plane* cortical clearance and which is then clipped to a central band of
    slices for endplate avoidance.

    Why in-plane + band-clip instead of a strict 3D sphere: CT slices are far
    thicker than the in-plane pixels (e.g. 2.5 mm vs 0.5 mm), so a symmetric 3D
    sphere with a fixed endplate margin gets choked by the through-slice
    direction and collapses on thinner vertebrae. Sizing the radius from the
    axial cortical clearance and avoiding endplates by staying within the
    central `central_height_frac` of body height keeps the ROI a robust,
    reproducible sample of mid-vertebral cancellous bone.
    """
    axes = compute_local_axes(body, spacing)
    central = central_height_mask(body, axes, spacing, params.central_height_frac)
    # center: body centroid, pulled into the central band / interior if needed
    dist3d = body_distance(body, spacing)
    center = _clamp_center_to_body(tuple(axes.centroid), body & central, dist3d * central)

    # In-plane cortical clearance at the center's axial slice (sampling uses the
    # two in-plane spacings). This is independent of slice thickness.
    sx, sy, sz = spacing
    slc = body[:, :, center[2]]
    if slc.any():
        inplane = distance_transform_edt(slc, sampling=(sx, sy))
        max_safe = float(inplane[center[0], center[1]])
    else:
        max_safe = 0.0

    margin = proportional_margin(max_safe, params.margin_floor_mm, params.margin_frac)
    radius = max(min(params.target_radius_mm,
                     params.radius_frac * max_safe,
                     max_safe - margin), 0.0)

    warnings = []
    if radius < params.min_radius_mm:
        warnings.append(f"safe radius {radius:.1f}mm below floor {params.min_radius_mm}mm")

    sphere = make_sphere(body.shape, center, radius, spacing)
    roi = sphere & body & central                 # clip to bone + endplate-safe band
    band_sphere_vox = int((sphere & central).sum())   # intended shape (for QC clipping)
    return {
        "mode": "centroid_sphere",
        "roi_mask": roi,
        "center_idx": center,
        "radius_mm": radius,
        "max_safe_radius_mm": max_safe,
        "margin_mm": max_safe - radius,   # in-plane cortical clearance
        "band_sphere_vox": band_sphere_vox,
        "axes": axes,
        "warnings": warnings,
    }


def centroid_volume_sphere(body, spacing, params: ROIParams, hu=None):
    """Default measurement ROI: a volume-proportional sphere anchored at the
    vertebral-body centroid.

    This is the project's reproducibility-first design. Placement is determined
    by **anatomy alone** -- the body centroid pulled into the central-height
    band -- so the ROI location never depends on the HU values being measured
    (unlike the lowest-attenuation search, which biases toward the softest spot).
    The radius is **proportional to body volume** (Westerhoff's size-scaling, so
    a small T-body and a large L-body get comparably representative samples) but
    then capped so the *entire* sphere stays inside the body with a margin from
    every surface -- cortex AND endplates. Because the sphere is fully contained
    it is NOT clipped per-plane, so it renders as a true, consistent sphere in
    all three views (no flattened lens in sagittal/coronal) and never touches an
    endplate. The reported statistic is the median HU (computed downstream).
    """
    axes = compute_local_axes(body, spacing)
    central = central_height_mask(body, axes, spacing, params.central_height_frac)
    dist = body_distance(body, spacing)
    center = _clamp_center_to_body(tuple(axes.centroid), body & central, dist * central)

    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    v_body = float(body.sum()) * voxel_vol
    r_target = (3.0 / (4.0 * np.pi) * params.roi_volume_frac * v_body) ** (1.0 / 3.0)
    r_target = float(min(r_target, params.target_radius_mm))

    # 3D clearance from the center to the nearest body surface (cortex OR
    # endplate); keep a proportional margin so the full sphere stays interior.
    max_safe = float(dist[center])
    margin = proportional_margin(max_safe, params.margin_floor_mm, params.margin_frac)
    r = float(max(min(r_target, max_safe - margin), 0.0))

    warnings = []
    if r < params.min_radius_mm:
        warnings.append(f"endplate-safe radius {r:.1f}mm below floor "
                        f"{params.min_radius_mm}mm")

    sphere = make_sphere(body.shape, center, r, spacing)
    roi = sphere & body          # r <= clearance, so this is a full (unclipped) sphere
    return {
        "mode": "centroid_volume_sphere",
        "roi_mask": roi,
        "center_idx": center,
        "radius_mm": r,
        "max_safe_radius_mm": max_safe,
        "margin_mm": max_safe - r,        # 3D clearance from cortex/endplate
        "band_sphere_vox": int(sphere.sum()),
        "axes": axes,
        "warnings": warnings,
    }


def largest_safe_sphere(body, spacing, params: ROIParams, hu=None):
    """Sphere at the distance-transform maximum (anatomy-adaptive, less reproducible)."""
    axes = compute_local_axes(body, spacing)
    central = central_height_mask(body, axes, spacing, params.central_height_frac)
    margin = proportional_margin(float(body_distance(body, spacing).max()),
                                 params.margin_floor_mm, params.margin_frac)
    inner, dist = make_inner(body, spacing, margin)
    region = inner & central
    if not region.any():
        region = inner if inner.any() else body
    dist_inner = body_distance(region, spacing)
    center = tuple(int(v) for v in np.unravel_index(np.argmax(dist_inner), dist_inner.shape))
    max_safe = float(dist[center])
    radius = min(params.target_radius_mm, params.radius_frac * max_safe)
    roi = make_sphere(body.shape, center, radius, spacing) & inner
    return {
        "mode": "largest_safe_sphere",
        "roi_mask": roi,
        "center_idx": center,
        "radius_mm": radius,
        "max_safe_radius_mm": max_safe,
        "margin_mm": margin,
        "axes": axes,
        "warnings": [],
    }


def trabecular_core(body, spacing, params: ROIParams, hu=None):
    """Whole eroded inner compartment within the central band (volumetric)."""
    axes = compute_local_axes(body, spacing)
    margin = proportional_margin(float(body_distance(body, spacing).max()),
                                 params.margin_floor_mm, params.margin_frac)
    inner, dist = make_inner(body, spacing, margin)
    central = central_height_mask(body, axes, spacing, params.central_height_frac)
    roi = inner & central
    center = tuple(int(v) for v in axes.centroid)
    return {
        "mode": "trabecular_core",
        "roi_mask": roi,
        "center_idx": center,
        "radius_mm": float("nan"),
        "max_safe_radius_mm": float(dist.max()),
        "margin_mm": margin,
        "axes": axes,
        "warnings": [],
    }


def lowest_attenuation_sphere(body, spacing, params: ROIParams, hu=None):
    """Volume-proportional sphere placed at the lowest-attenuation interior spot
    anterior to the basivertebral foramen (Westerhoff et al.).

    The sphere volume is `roi_volume_frac` of the body volume; its center is
    searched over interior, central-band, anterior candidates to minimize mean
    attenuation -- this dodges bone islands / osteophytes that falsely raise HU.
    The reported statistic is the median HU (computed downstream).
    """
    axes = compute_local_axes(body, spacing)
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    v_body = float(body.sum()) * voxel_vol
    r = (3.0 / (4.0 * np.pi) * params.roi_volume_frac * v_body) ** (1.0 / 3.0)
    r = float(min(max(r, params.min_radius_mm), params.target_radius_mm))

    dist = body_distance(body, spacing)
    central = central_height_mask(body, axes, spacing, params.central_height_frac)
    anterior = anterior_mask(body, axes, spacing, params.anterior_frac)
    interior = dist >= r
    valid = interior & central & anterior
    for fallback in (interior & central, interior):
        if valid.any():
            break
        valid = fallback

    warnings = []
    if not valid.any():
        # body too thin for this radius anywhere: take the deepest point, shrink
        center = tuple(int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))
        r = float(dist[center])
        warnings.append("body too thin for target ROI; radius shrunk to fit")
    elif hu is not None:
        kernel = ball(r, spacing).astype(np.float32)
        num = convolve(np.where(body, hu, 0.0).astype(np.float32), kernel,
                       mode="constant", cval=0.0)
        den = convolve(body.astype(np.float32), kernel, mode="constant", cval=0.0)
        local_mean = np.full(body.shape, np.inf, dtype=np.float32)
        ok = den > 0
        local_mean[ok] = num[ok] / den[ok]
        local_mean[~valid] = np.inf
        center = tuple(int(v) for v in np.unravel_index(np.argmin(local_mean),
                                                        local_mean.shape))
    else:
        d2 = np.where(valid, dist, -1.0)
        center = tuple(int(v) for v in np.unravel_index(np.argmax(d2), d2.shape))

    if r < params.min_radius_mm:
        warnings.append(f"safe radius {r:.1f}mm below floor {params.min_radius_mm}mm")

    sphere = make_sphere(body.shape, center, r, spacing)
    roi = sphere & body
    return {
        "mode": "lowest_attenuation_sphere",
        "roi_mask": roi,
        "center_idx": center,
        "radius_mm": r,
        "max_safe_radius_mm": float(dist[center]),
        "margin_mm": float(dist[center]) - r,
        "band_sphere_vox": int(sphere.sum()),
        "axes": axes,
        "warnings": warnings,
    }


def axial_ellipse_2d(body, spacing, params: ROIParams, hu=None):
    """Single mid-vertebral axial elliptical ROI (Jang et al. 2D comparator)."""
    axes = compute_local_axes(body, spacing)
    central = central_height_mask(body, axes, spacing, params.central_height_frac)
    dist3d = body_distance(body, spacing)
    center = _clamp_center_to_body(tuple(axes.centroid), body & central,
                                   dist3d * central)
    sx, sy, sz = spacing
    slc = body[:, :, center[2]]
    if slc.any():
        inplane = distance_transform_edt(slc, sampling=(sx, sy))
        max_safe = float(inplane[center[0], center[1]])
    else:
        max_safe = 0.0
    margin = proportional_margin(max_safe, params.margin_floor_mm, params.margin_frac)
    radius = max(min(params.target_radius_mm, params.radius_frac * max_safe,
                     max_safe - margin), 0.0)

    roi = np.zeros_like(body)
    if radius > 0 and slc.any():
        X, Y = np.meshgrid(np.arange(body.shape[0]), np.arange(body.shape[1]),
                           indexing="ij")
        d2 = ((X - center[0]) * sx) ** 2 + ((Y - center[1]) * sy) ** 2
        roi[:, :, center[2]] = (d2 <= radius ** 2) & slc

    return {
        "mode": "axial_ellipse_2d",
        "roi_mask": roi,
        "center_idx": center,
        "radius_mm": radius,
        "max_safe_radius_mm": max_safe,
        "margin_mm": max_safe - radius,
        "band_sphere_vox": int(roi.sum()),
        "axes": axes,
        "warnings": [],
    }


ROI_MODES = {
    "centroid_volume_sphere": centroid_volume_sphere,
    "lowest_attenuation_sphere": lowest_attenuation_sphere,
    "centroid_sphere": centroid_sphere,
    "largest_safe_sphere": largest_safe_sphere,
    "trabecular_core": trabecular_core,
    "axial_ellipse_2d": axial_ellipse_2d,
}

# Reproducible-by-design default: centroid placement + volume-proportional size.
DEFAULT_MODE = "centroid_volume_sphere"

# ROI methods compared head-to-head for the reproducibility / variance study:
# our default centroid + volume-proportional 3D, Westerhoff's lowest-attenuation
# 3D, and the classic Jang 2D single-slice ROI.
COMPARISON_MODES = ["centroid_volume_sphere", "lowest_attenuation_sphere",
                    "axial_ellipse_2d"]


def place_roi(body, spacing, params: ROIParams | None = None,
              mode: str = DEFAULT_MODE, hu=None) -> dict:
    params = params or ROIParams()
    if mode not in ROI_MODES:
        raise ValueError(f"unknown ROI mode: {mode}")
    return ROI_MODES[mode](body, spacing, params, hu)
