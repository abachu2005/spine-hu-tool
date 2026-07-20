"""Run library tests: save/list/round-trip, overrides, batch, and reopen.

Reopen is exercised with the DICOM load and segmentation STUBBED -- this never
runs real local segmentation.
"""
import os

import numpy as np
import pytest

from spine_hu_tool.core import Volume
from spine_hu_tool.app.review_state import ReviewState
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from spine_hu_tool.app import run_store as rs
from .synthetic import make_vertebra_phantom


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINE_HU_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _phantom_state():
    full, _body, hu = make_vertebra_phantom()
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    vol = Volume(hu=hu, spacing=(1.0, 1.0, 1.0))
    return ReviewState.from_volume(vol, seg, levels=["L1"]), vol, seg


def _meta(files):
    return {
        "patient_id": "P1", "patient_label": "P1", "study_uid": "stA",
        "study_description": "CT", "study_date": "20240101",
        "series_uid": "1.2.3", "series_description": "d", "n_files": len(files),
        "source_folder": os.path.dirname(files[0]) if files else "",
        "files": list(files),
    }


def test_save_load_list_roundtrip(runs_dir):
    state, _v, _s = _phantom_state()
    rec = rs.build_record(state, _meta(["/x/a.dcm"]), backend="cloud",
                          resolution="full", mode="centroid_volume_sphere")
    rs.save_run(rec)
    runs = rs.list_runs()
    assert len(runs) == 1 and runs[0]["id"] == rec["id"]
    assert rs.load_run(rec["id"])["study"]["series_uid"] == "1.2.3"


def test_overrides_roundtrip(runs_dir):
    state, _v, seg = _phantom_state()
    state.set_radius("L1", 6.5)
    state.set_decision("L1", True)
    rec = rs.build_record(state, _meta(["/x/a.dcm"]), backend="local",
                          resolution="fast", mode="centroid_volume_sphere")
    rs.save_run(rec)
    ov = rs.load_run(rec["id"])["overrides"]["L1"]
    assert ov["accepted"] is True and abs(ov["radius_mm"] - 6.5) < 1e-6

    # replay onto a fresh state
    full, _b, hu = make_vertebra_phantom()
    state2 = ReviewState.from_volume(Volume(hu=hu, spacing=(1.0, 1.0, 1.0)),
                                     seg, levels=["L1"])
    rs.apply_overrides(state2, {"L1": ov})
    assert state2.results["L1"].accepted is True
    assert abs(state2.results["L1"].radius_mm - 6.5) < 1e-6


def test_batch_save_and_list(runs_dir):
    state, _v, _s = _phantom_state()
    ids = []
    for _ in range(2):
        rec = rs.build_record(state, _meta(["/x/a.dcm"]), backend="cloud",
                              resolution="full", mode="centroid_volume_sphere")
        rs.save_run(rec)
        ids.append(rec["id"])
    batch = rs.new_batch(ids, source_folder="/x")
    rs.save_batch(batch)
    listed = rs.list_batches()
    assert len(listed) == 1 and listed[0]["run_ids"] == ids


def test_failed_record(runs_dir):
    rec = rs.failed_record(_meta(["/x/a.dcm"]), "boom", backend="local",
                           resolution="full", mode="centroid_volume_sphere")
    rs.save_run(rec)
    loaded = rs.load_run(rec["id"])
    assert loaded["status"] == "failed" and loaded["error"] == "boom"


def test_reopen_rebuilds_state_with_stubbed_seg(runs_dir, tmp_path, monkeypatch):
    state, vol, seg = _phantom_state()
    state.set_radius("L1", 7.0)
    state.set_decision("L1", True)

    # real files on disk so the reopen file-existence check passes
    dcm = tmp_path / "a.dcm"
    dcm.write_bytes(b"stub")
    rec = rs.build_record(state, _meta([str(dcm)]), backend="local",
                          resolution="full", mode="centroid_volume_sphere")
    rs.save_run(rec)

    # a cached seg file that reopen will find (content irrelevant; load stubbed)
    seg_file = tmp_path / "seg.nii.gz"
    seg_file.write_bytes(b"stub")

    import spine_hu_tool.io.dicom_loader as dl
    import spine_hu_tool.segmentation.totalseg_runner as tr
    import spine_hu_tool.app.analysis as an
    monkeypatch.setattr(dl, "load_series", lambda files, metadata=None: vol)
    monkeypatch.setattr(tr, "load_segmentation", lambda p: seg)
    monkeypatch.setattr(an, "cached_seg_path", lambda uid: str(seg_file))

    reopened = rs.reopen_run(rec["id"])
    assert reopened.results["L1"].accepted is True
    assert abs(reopened.results["L1"].radius_mm - 7.0) < 1e-6


def test_reopen_missing_source_raises(runs_dir, tmp_path, monkeypatch):
    state, _v, _s = _phantom_state()
    rec = rs.build_record(state, _meta(["/nonexistent/a.dcm"]), backend="local",
                          resolution="full", mode="centroid_volume_sphere")
    # no source_folder that exists either
    rec["study"]["source_folder"] = "/also/missing"
    rs.save_run(rec)
    with pytest.raises(rs.RunReopenError):
        rs.reopen_run(rec["id"])


def test_reopen_missing_seg_cache_raises(runs_dir, tmp_path, monkeypatch):
    state, vol, _s = _phantom_state()
    dcm = tmp_path / "a.dcm"
    dcm.write_bytes(b"stub")
    rec = rs.build_record(state, _meta([str(dcm)]), backend="local",
                          resolution="full", mode="centroid_volume_sphere")
    rs.save_run(rec)

    import spine_hu_tool.io.dicom_loader as dl
    import spine_hu_tool.app.analysis as an
    monkeypatch.setattr(dl, "load_series", lambda files, metadata=None: vol)
    monkeypatch.setattr(an, "cached_seg_path", lambda uid: str(tmp_path / "missing_seg.nii.gz"))
    with pytest.raises(rs.RunReopenError):
        rs.reopen_run(rec["id"])
