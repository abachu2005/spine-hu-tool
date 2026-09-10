import numpy as np
from spine_hu_tool.core import Volume
from spine_hu_tool.app.review_state import ReviewState
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import make_vertebra_phantom


def _synthetic_state(tmp_path=None, spacing=(1.0, 1.0, 1.0),
                     compute_comparison=False):
    full, body, hu = make_vertebra_phantom(spacing=spacing)
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    vol = Volume(hu=hu, spacing=spacing)
    audit = str(tmp_path / "audit.json") if tmp_path else None
    return ReviewState.from_volume(
        vol, seg, levels=["L1"], audit_path=audit,
        compute_comparison=compute_comparison)


def _failing_state(tmp_path=None):
    full, _body, hu = make_vertebra_phantom()
    hu[full] = 3000.0
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    vol = Volume(hu=hu, spacing=(1.0, 1.0, 1.0))
    audit = str(tmp_path / "audit.json") if tmp_path else None
    return ReviewState.from_volume(vol, seg, levels=["L1"], audit_path=audit)


def test_set_radius_recomputes_and_audits(tmp_path):
    st = _synthetic_state(tmp_path)
    r = st.results["L1"]
    before = r.stats["mean_HU"]
    st.set_radius("L1", 2.5)
    assert abs(r.radius_mm - 2.5) < 1e-6
    assert r.stats["mean_HU"] is not None
    assert any(e["action"] == "set_radius" for e in st.audit.entries)


def test_decision_and_serialize(tmp_path):
    st = _synthetic_state(tmp_path)
    st.set_decision("L1", True)
    assert st.results["L1"].accepted is True
    ser = st.serialize()
    assert "L1" in ser["results"]
    assert ser["results"]["L1"]["accepted"] is True
    assert ser["results"]["L1"]["included"] is True


def test_failed_level_is_editable_but_excluded_until_explicitly_included(tmp_path):
    st = _failing_state(tmp_path)
    r = st.results["L1"]
    assert r.qc["qc_status"] == "fail"
    assert r.editable
    assert r.accepted is None
    assert not r.included
    assert r.stats["median_HU"] == 3000.0

    st.set_radius("L1", 4.0)
    assert st.results["L1"].accepted is None
    assert not st.results["L1"].included

    st.recompute_level("L1")
    assert st.results["L1"].accepted is None
    assert not st.results["L1"].included

    st.set_decision("L1", True)
    assert st.results["L1"].included
    assert st.audit.entries[-1]["action"] == "include"

    st.set_radius("L1", 4.5)
    assert st.results["L1"].accepted is None
    assert not st.results["L1"].included


def test_passing_level_can_be_manually_excluded():
    st = _synthetic_state()
    assert st.results["L1"].included
    st.set_decision("L1", False)
    assert not st.results["L1"].included
    assert st.results["L1"].exclusion_reason() == "manually excluded by reviewer"


def test_nudge_center_moves_roi(tmp_path):
    st = _synthetic_state(tmp_path)
    c0 = st.results["L1"].center_idx
    st.nudge_center("L1", (2, 0, 0))
    assert st.results["L1"].center_idx[0] == c0[0] + 2


def test_center_outside_body_is_rejected():
    st = _synthetic_state()
    r = st.results["L1"]
    original = r.center_idx
    st.set_center("L1", (0, 0, 0))
    assert r.center_idx == original
    assert r.editable


def test_recompute_resets_edit(tmp_path):
    st = _synthetic_state(tmp_path)
    st.set_radius("L1", 2.5)
    auto = st.recompute_level("L1")
    assert auto.radius_mm > 2.5    # back to automatic adaptive radius


def test_edits_preserve_annotations_and_invalidate_stale_comparison():
    st = _synthetic_state(compute_comparison=True)
    r = st.results["L1"]
    assert r.comparison is not None
    r.stats.update({
        "body_width_lr_mm": 380.0,
        "scout_ap_magnification": 1.02,
    })
    warning = "scout: synthetic outline warning"
    r.qc.setdefault("warnings", []).append(warning)
    r.qc.setdefault("persistent_warnings", []).append(warning)
    r.qc["level_status"] = "review"
    r.qc["qc_status"] = "review"

    st.set_radius("L1", 4.0)
    r = st.results["L1"]
    assert r.stats["body_width_lr_mm"] == 380.0
    assert r.stats["scout_ap_magnification"] == 1.02
    assert "calibrated_median_HU" in r.stats
    assert "hu_context" in r.stats
    assert warning in r.qc["warnings"]
    assert r.qc["qc_status"] == "review"
    assert r.comparison is None

    r = st.recompute_level("L1")
    assert r.stats["body_width_lr_mm"] == 380.0
    assert r.stats["scout_ap_magnification"] == 1.02
    assert "calibrated_median_HU" in r.stats
    assert "hu_context" in r.stats
    assert warning in r.qc["warnings"]
    assert r.qc["qc_status"] == "review"
    assert r.comparison is not None


def test_real_scout_warning_survives_roi_edit(monkeypatch):
    from spine_hu_tool.app.analysis import measure_scout
    from spine_hu_tool.scout import thickness

    st = _synthetic_state()
    warning = "scout outline clipped"
    monkeypatch.setattr(
        thickness, "habitus_for_case",
        lambda *args, **kwargs: {
            "available": True,
            "levels": {
                "L1": {
                    "body_width_lr_mm": 380.0,
                    "scout_warnings": [warning],
                }
            },
        })
    measure_scout(st, "/unused")
    st.set_radius("L1", 4.0)
    assert warning in st.results["L1"].qc["warnings"]
