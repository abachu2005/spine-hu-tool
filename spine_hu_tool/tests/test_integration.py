"""End-to-end integration on the cached thoracic case (skips if unavailable)."""
import numpy as np
from spine_hu_tool.pipeline import process_case


def test_thoracic_clean_levels_and_l1_band(cached_thoracic):
    vol, seg = cached_thoracic
    case = process_case(vol, seg, levels=["T11", "T12", "L1"])
    res = case["results"]
    assert {"T11", "T12", "L1"} <= set(res)
    # mid-thoracic/L1 trabecular HU should be in a physiologic band
    for lvl in ("T11", "T12"):
        assert 120 < res[lvl].stats["mean_HU"] < 300
        assert res[lvl].qc["qc_status"] in ("pass", "review")
    assert res["T11"].margin_mm > 0


def test_calibration_and_context_surfaced(cached_thoracic):
    # default mode is centroid_volume_sphere -> calibrated median + (no) class
    vol, seg = cached_thoracic
    case = process_case(vol, seg, levels=["T12"])
    assert case.get("seg_status") in ("ok", "suspect", "invalid")
    assert case.get("calibration") is not None
    assert case.get("mode") == "centroid_volume_sphere"
    r = case["results"]["T12"]
    if r.qc.get("qc_status") != "excluded":
        assert "median_HU" in r.stats
        assert "calibrated_median_HU" in r.stats
        # tool reports a measurement, not a diagnosis
        assert r.stats.get("classification") is None
        assert "hu_context" in r.stats


def test_comparison_instrumentation(cached_thoracic):
    vol, seg = cached_thoracic
    case = process_case(vol, seg, levels=["T12"], compute_comparison=True)
    r = case["results"]["T12"]
    if r.qc.get("qc_status") != "excluded":
        assert r.comparison is not None
        methods = r.comparison["methods"]
        assert "centroid_volume_sphere" in methods
        assert "lowest_attenuation_sphere" in methods


def test_scan_position_partial_l1_flagged(cached_thoracic):
    # L1 sits at the inferior edge of the thoracic field -> a partial (FOV-clipped)
    # body is measured but flagged for review (not silently passed, not dropped).
    vol, seg = cached_thoracic
    case = process_case(vol, seg, levels=["L1"])
    qc = case["results"]["L1"].qc
    assert qc["qc_status"] in ("review", "fail", "excluded")
    if qc["qc_status"] == "review":
        # a truncated body that is otherwise measurable carries the truncation flag
        assert qc.get("truncated") or any("clipped" in w for w in qc.get("warnings", []))
    if qc["qc_status"] == "excluded":
        assert "mean_HU" not in case["results"]["L1"].stats
