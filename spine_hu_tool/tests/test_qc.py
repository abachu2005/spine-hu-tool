import numpy as np
from spine_hu_tool.config import ROIParams
from spine_hu_tool.roi.modes import place_roi
from spine_hu_tool.measurement.hu_stats import compute_hu_stats
from spine_hu_tool.measurement.qc import compute_qc
from .synthetic import make_vertebra_phantom


def _qc_for(hu, body, info, params=None):
    params = params or ROIParams()
    stats = compute_hu_stats(hu, info["roi_mask"], (1.0, 1.0, 1.0))
    return compute_qc(hu, body, info, (1.0, 1.0, 1.0), stats, params), stats


def test_clean_phantom_passes():
    full, body, hu = make_vertebra_phantom()
    info = place_roi(body, (1.0, 1.0, 1.0), ROIParams(), "centroid_sphere")
    qc, _ = _qc_for(hu, body, info)
    assert qc["qc_status"] == "pass"
    assert not qc["near_metal"]


def test_metal_proximity_fails():
    full, body, hu = make_vertebra_phantom()
    info = place_roi(body, (1.0, 1.0, 1.0), ROIParams(), "centroid_sphere")
    c = info["center_idx"]
    hu[c[0] + 2, c[1], c[2]] = 3000.0   # metal just inside ROI
    qc, _ = _qc_for(hu, body, info)
    assert qc["near_metal"] and qc["qc_status"] == "fail"


def test_heterogeneity_flagged():
    full, body, hu = make_vertebra_phantom()
    info = place_roi(body, (1.0, 1.0, 1.0), ROIParams(), "centroid_sphere")
    roi = info["roi_mask"]
    idx = np.argwhere(roi)
    half = idx[: len(idx) // 2]
    hu[half[:, 0], half[:, 1], half[:, 2]] = 600.0   # inject spread
    qc, stats = _qc_for(hu, body, info)
    assert any("heterogeneity" in w for w in qc["warnings"])
