"""CLI helpers: the segmentation-status summary must always explain itself."""
from spine_hu_tool.app.cli import _seg_status_reasons


def test_reasons_prefers_global():
    case = {"seg_check": {"global_reasons": ["3 SI-ordering inversion(s)"],
                          "levels": {}}}
    assert _seg_status_reasons(case) == ["3 SI-ordering inversion(s)"]


def test_reasons_fall_back_to_per_level_when_no_global():
    # the Anon2 case: 'suspect' driven purely by bad levels, no global reason.
    case = {"seg_check": {"global_reasons": [],
                          "levels": {
                              "T8": {"valid": False,
                                     "reasons": ["implausibly small (219 mm3)"]},
                              "sacrum": {"valid": False,
                                         "reasons": ["implausibly large",
                                                     "SI overlap with non-adjacent S1"]},
                              "L1": {"valid": True, "reasons": []},
                          }}}
    reasons = _seg_status_reasons(case)
    assert any("T8" in r and "small" in r for r in reasons)
    assert any("sacrum" in r for r in reasons)
    assert all("L1" not in r for r in reasons)          # valid levels not listed


def test_reasons_empty_when_nothing_to_explain():
    assert _seg_status_reasons({}) == []
    assert _seg_status_reasons({"seg_check": {"global_reasons": [],
                                              "levels": {}}}) == []
