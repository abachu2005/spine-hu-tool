import csv
import json
import numpy as np
from spine_hu_tool.core import Volume
from spine_hu_tool.pipeline import process_case
from spine_hu_tool.export.writers import export_case
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import make_vertebra_phantom


def _case(spacing=(1.0, 1.0, 1.0)):
    full, body, hu = make_vertebra_phantom(spacing=spacing)
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    vol = Volume(hu=hu, spacing=spacing)
    return process_case(vol, seg, levels=["L1"]), vol


def _failed_case():
    full, _body, hu = make_vertebra_phantom()
    hu[full] = 3000.0
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    vol = Volume(hu=hu, spacing=(1.0, 1.0, 1.0))
    return process_case(vol, seg, levels=["L1"]), vol


def test_export_writes_all_artifacts(tmp_path):
    case, vol = _case()
    out = str(tmp_path / "out")
    written = export_case(case, vol, out)
    for key in ("csv", "json", "reproducibility", "overlays", "masks"):
        assert key in written

    payload = json.load(open(written["json"]))
    assert "L1" in payload["results"]


def test_csv_includes_data_and_decision(tmp_path):
    # the exported measurements CSV carries the per-level data including the
    # physician accept/reject decision.
    case, vol = _case()
    case["results"]["L1"].accepted = True
    out = str(tmp_path / "out")
    written = export_case(case, vol, out, overlays=False, masks=False)
    with open(written["csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["level"] == "L1"
    assert "median_HU" in rows[0] and "qc_status" in rows[0]
    assert rows[0]["accepted"] == "True"
    assert rows[0]["included"] == "True"


def test_excluded_result_is_sanitized_but_reproducibility_keeps_raw_measurement(tmp_path):
    case, vol = _failed_case()
    r = case["results"]["L1"]
    assert not r.included and "median_HU" in r.stats

    written = export_case(
        case, vol, str(tmp_path / "out"), overlays=False, masks=False)
    payload = json.load(open(written["json"]))
    final = payload["results"]["L1"]
    assert final["included"] is False
    assert "median_HU" not in final
    assert final["exclusion_reason"]

    with open(written["csv"], newline="") as f:
        row = list(csv.DictReader(f))[0]
    assert row["included"] == "False"
    assert "median_HU" not in row

    rec = json.load(open(written["reproducibility"]))
    raw = rec["measurements"]["L1"]
    assert raw["included"] is False
    assert raw["median_HU"] == 3000.0
    assert rec["inclusion_policy"] == "qc-fail-default-excluded-explicit-include"


def test_reproducibility_matches_measurement(tmp_path):
    case, vol = _case()
    out = str(tmp_path / "out")
    written = export_case(case, vol, out, overlays=False, masks=False)
    rec = json.load(open(written["reproducibility"]))
    measured = case["results"]["L1"].stats["mean_HU"]
    assert abs(rec["measurements"]["L1"]["mean_HU"] - round(measured, 2)) < 0.01
