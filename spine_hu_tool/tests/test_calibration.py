"""kVp/scanner calibration factors, soft L1 context, and Table 6 thresholds."""
from spine_hu_tool.measurement.calibration import (compute_calibration, classify,
                                                   reference_context)


def test_kvp_80_factor():
    cal = compute_calibration({"kvp": 80, "manufacturer_model": "Mystery CT"})
    assert abs(cal["kvp_factor"] - 0.773) < 1e-6
    assert cal["scanner_factor"] == 1.0           # unknown scanner -> no correction
    assert abs(cal["factor"] - 0.773) < 1e-6
    assert cal["calibrated"] is True


def test_unknown_acquisition_factor_one():
    cal = compute_calibration({})
    assert cal["factor"] == 1.0
    assert cal["calibrated"] is False


def test_reference_scanner_at_120kvp():
    cal = compute_calibration({"kvp": 120,
                               "manufacturer_model": "Siemens Somatom Definition Edge"})
    assert cal["factor"] == 1.0


def test_classification_matches_table6():
    # `classify` is retained (tested) but is NOT used for the default measurement;
    # L1 thresholds: (osteoporosis 80, osteopenia 117)
    assert classify("L1", 70) == "osteoporosis"
    assert classify("L1", 100) == "osteopenia"
    assert classify("L1", 130) == "normal"
    assert classify("L1", None) is None
    assert classify("UNKNOWN", 100) is None


def test_reference_context_is_l1_only_and_non_diagnostic():
    # soft Pickhardt anchors (osteoporosis ~110, normal ~160), context only
    assert reference_context("L1", 90) is not None and "below" in reference_context("L1", 90)
    assert "between" in reference_context("L1", 130)
    assert "above" in reference_context("L1", 200)
    # other levels get no anchor; missing HU -> none
    assert reference_context("T12", 130) is None
    assert reference_context("L1", None) is None
    # explicitly never returns a diagnostic class word
    for hu in (90, 130, 200):
        assert "osteoporosis:" not in (reference_context("L1", hu) or "")
