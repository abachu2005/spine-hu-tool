import numpy as np
from spine_hu_tool.config import ROIParams
from spine_hu_tool.roi.distance import make_inner, body_distance
from spine_hu_tool.roi.modes import place_roi
from spine_hu_tool.measurement.hu_stats import compute_hu_stats
from .synthetic import make_vertebra_phantom


def test_inner_respects_margin():
    full, body_true, _hu = make_vertebra_phantom()
    inner, dist = make_inner(body_true, (1.0, 1.0, 1.0), margin_mm=3.0)
    # every inner voxel is at least margin from the boundary
    assert dist[inner].min() >= 3.0
    assert inner.sum() > 0


def test_centroid_sphere_clears_cortex_and_measures_trabecular():
    full, body_true, hu = make_vertebra_phantom()
    info = place_roi(body_true, (1.0, 1.0, 1.0), ROIParams(), "centroid_sphere")
    stats = compute_hu_stats(hu, info["roi_mask"], (1.0, 1.0, 1.0))
    # ROI sits in trabecular bone (~150), not cortex (700)
    assert 120 < stats["mean_HU"] < 220
    assert info["margin_mm"] > 0
    assert info["radius_mm"] >= 3.0


def test_reproducible_across_spacings():
    means = {}
    for sp in [(1.0, 1.0, 1.0), (1.5, 1.5, 1.5)]:
        full, body_true, hu = make_vertebra_phantom(spacing=sp)
        info = place_roi(body_true, sp, ROIParams(), "centroid_sphere")
        means[sp] = compute_hu_stats(hu, info["roi_mask"], sp)["mean_HU"]
    assert abs(means[(1.0, 1.0, 1.0)] - means[(1.5, 1.5, 1.5)]) < 15.0


def test_modes_available():
    full, body_true, hu = make_vertebra_phantom()
    for mode in ("centroid_sphere", "largest_safe_sphere", "trabecular_core"):
        info = place_roi(body_true, (1.0, 1.0, 1.0), ROIParams(), mode)
        assert info["roi_mask"].sum() > 0


def test_cylinder_volume_trabecular_and_contained():
    full, body_true, hu = make_vertebra_phantom()
    sp = (1.0, 1.0, 1.0)
    info = place_roi(body_true, sp, ROIParams(), "cylinder_volume")
    roi = info["roi_mask"]
    assert roi.sum() > 0
    # the whole cylinder stays inside the body (cortex/endplate clearance)
    assert int((roi & ~body_true).sum()) == 0
    stats = compute_hu_stats(hu, roi, sp)
    assert 120 < stats["mean_HU"] < 220        # trabecular bone, not cortex
    assert info["radius_mm"] > 0
    assert info["height_mm"] > 0


def test_centroid_center_is_concentric_with_cross_section():
    """The mid-body center must sit at the cross-section center so the ROI
    renders concentric in the sagittal/coronal views (centering fix)."""
    full, body_true, hu = make_vertebra_phantom()
    sp = (1.0, 1.0, 1.0)
    info = place_roi(body_true, sp, ROIParams(), "centroid_volume_sphere")
    cx, cy, cz = info["center_idx"]
    coords = np.argwhere(body_true[:, :, cz])      # (x, y) of the mid-axial slice
    mean_x, mean_y = coords.mean(axis=0)
    assert abs(mean_x - cx) <= 3
    assert abs(mean_y - cy) <= 3
