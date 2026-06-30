"""ROI modes: lowest-attenuation avoids bone islands; 2D comparator works."""
import numpy as np

from spine_hu_tool.roi.body_isolation import isolate_body
from spine_hu_tool.roi.modes import place_roi
from spine_hu_tool.config import ROIParams
from spine_hu_tool.geometry.coords import make_sphere
from spine_hu_tool.measurement.hu_stats import compute_hu_stats
from .synthetic import make_vertebra_phantom, _phys_grid, _ellipsoid

SPACING = (1.0, 1.0, 1.0)


def test_lowest_attenuation_avoids_bone_island():
    full, body_true, hu = make_vertebra_phantom()
    hu = hu.copy()
    X, Y, Z = _phys_grid(hu.shape, SPACING)
    island = _ellipsoid(X, Y, Z, (30, 22, 25), (5, 5, 5))  # dense, at centroid
    hu[island & body_true] = 1200.0

    body, _flags = isolate_body(full, SPACING)
    params = ROIParams()
    la = place_roi(body, SPACING, params, "lowest_attenuation_sphere", hu=hu)
    cen = place_roi(body, SPACING, params, "centroid_sphere", hu=hu)

    # the sphere is fully interior to the body (no cortical clipping)
    sphere = make_sphere(body.shape, la["center_idx"], la["radius_mm"], SPACING)
    assert int((sphere & ~body).sum()) == 0

    # ROI volume is ~5% of the body volume
    frac = float(la["roi_mask"].sum()) / float(body.sum())
    assert 0.02 < frac < 0.09

    # lowest-attenuation ROI steers clear of the dense island that the centered
    # ROI sits on top of
    la_overlap = int((la["roi_mask"] & island).sum())
    cen_overlap = int((cen["roi_mask"] & island).sum())
    island_vox = int(island.sum())
    assert la_overlap < cen_overlap
    assert la_overlap < 0.02 * island_vox      # essentially avoids it

    # and so reads a lower (cleaner trabecular) attenuation
    la_mean = compute_hu_stats(hu, la["roi_mask"], SPACING)["mean_HU"]
    cen_mean = compute_hu_stats(hu, cen["roi_mask"], SPACING)["mean_HU"]
    assert la_mean < cen_mean


def test_centroid_volume_sphere_is_hu_independent_and_volume_scaled():
    # Default mode: placement must NOT move when HU content changes (reproducible
    # by design), and the radius must scale with body volume (~5%).
    full, body_true, hu = make_vertebra_phantom()
    body, _flags = isolate_body(full, SPACING)
    params = ROIParams()

    base = place_roi(body, SPACING, params, "centroid_volume_sphere", hu=hu)

    # drop a dense island right at the centroid: a lowest-attenuation search
    # would flee it, but the centroid anchor must stay put.
    hu2 = hu.copy()
    X, Y, Z = _phys_grid(hu.shape, SPACING)
    island = _ellipsoid(X, Y, Z, (30, 22, 25), (5, 5, 5))
    hu2[island & body_true] = 1500.0
    moved = place_roi(body, SPACING, params, "centroid_volume_sphere", hu=hu2)

    assert base["center_idx"] == moved["center_idx"]          # HU-independent
    assert abs(base["radius_mm"] - moved["radius_mm"]) < 1e-9

    frac = float(base["roi_mask"].sum()) / float(body.sum())
    assert 0.02 < frac < 0.09                                  # volume-proportional
    assert base["radius_mm"] <= params.target_radius_mm

    # the sphere is fully contained in the body (true sphere, no per-plane
    # clipping) and clears every surface -> renders consistently in all views
    sphere = make_sphere(body.shape, base["center_idx"], base["radius_mm"], SPACING)
    assert int((sphere & ~body).sum()) == 0
    assert int((base["roi_mask"] ^ (sphere & body)).sum()) == 0
    assert base["margin_mm"] > 0                               # clearance from cortex/endplate


def test_axial_ellipse_single_slice_value():
    full, _body_true, hu = make_vertebra_phantom()
    body, _flags = isolate_body(full, SPACING)
    info = place_roi(body, SPACING, ROIParams(), "axial_ellipse_2d", hu=hu)
    assert info["radius_mm"] > 0
    s = compute_hu_stats(hu, info["roi_mask"], SPACING)
    assert s["voxel_count"] > 0
    zs = np.unique(np.argwhere(info["roi_mask"])[:, 2])
    assert len(zs) == 1            # truly a single axial slice
