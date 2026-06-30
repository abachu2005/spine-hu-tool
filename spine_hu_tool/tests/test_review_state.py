import numpy as np
from spine_hu_tool.core import Volume
from spine_hu_tool.app.review_state import ReviewState
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import make_vertebra_phantom


def _synthetic_state(tmp_path=None, spacing=(1.0, 1.0, 1.0)):
    full, body, hu = make_vertebra_phantom(spacing=spacing)
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    vol = Volume(hu=hu, spacing=spacing)
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


def test_nudge_center_moves_roi(tmp_path):
    st = _synthetic_state(tmp_path)
    c0 = st.results["L1"].center_idx
    st.nudge_center("L1", (2, 0, 0))
    assert st.results["L1"].center_idx[0] == c0[0] + 2


def test_recompute_resets_edit(tmp_path):
    st = _synthetic_state(tmp_path)
    st.set_radius("L1", 2.5)
    auto = st.recompute_level("L1")
    assert auto.radius_mm > 2.5    # back to automatic adaptive radius
