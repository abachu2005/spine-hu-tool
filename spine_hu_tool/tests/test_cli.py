"""CLI helpers: screening and inclusion state must print safely."""
import numpy as np

from spine_hu_tool.app.cli import _seg_status_reasons, _print_summary
from spine_hu_tool.core import Volume
from spine_hu_tool.pipeline import process_case
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import make_vertebra_phantom


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


def test_summary_does_not_print_hu_for_excluded_result(capsys):
    full, _body, hu = make_vertebra_phantom()
    hu[full] = 3000.0
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    case = process_case(
        Volume(hu=hu, spacing=(1.0, 1.0, 1.0)), seg, levels=["L1"])

    _print_summary(case)
    output = capsys.readouterr().out
    level_row = next(line for line in output.splitlines()
                     if line.startswith("L1"))
    assert "EXCLUDED:" in level_row
    assert level_row.split()[1:3] == ["--", "--"]
