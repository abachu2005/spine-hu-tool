"""End-to-end inclusion policy across screening, measurement, and review."""
import numpy as np
import pytest

from spine_hu_tool.core import Volume
from spine_hu_tool.pipeline import process_case
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import make_vertebra_phantom


def _volume_and_seg():
    full, _body, hu = make_vertebra_phantom()
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    return Volume(hu=hu, spacing=(1.0, 1.0, 1.0)), seg


def _seg_check(valid=True):
    return {
        "seg_status": "ok" if valid else "suspect",
        "global_reasons": [],
        "levels": {
            "L1": {
                "valid": valid,
                "reasons": [] if valid else ["synthetic geometry concern"],
                "voxels": 10000,
            }
        },
    }


def test_segmentation_invalid_level_keeps_editable_measurement(monkeypatch):
    import spine_hu_tool.pipeline as pipeline

    volume, seg = _volume_and_seg()
    monkeypatch.setattr(pipeline, "check_segmentation", lambda *_: _seg_check(False))
    monkeypatch.setattr(
        pipeline, "tag_levels",
        lambda *_: [{"level": "L1", "status": "clean"}])

    r = process_case(volume, seg, levels=["L1"])["results"]["L1"]
    assert r.editable and r.stats
    assert not r.included and r.accepted is None
    assert r.qc["auto_excluded"] is True
    assert "segmentation invalid" in r.qc["exclusion_reason"]


def test_instrumented_level_keeps_editable_measurement(monkeypatch):
    import spine_hu_tool.pipeline as pipeline

    volume, seg = _volume_and_seg()
    monkeypatch.setattr(pipeline, "check_segmentation", lambda *_: _seg_check(True))
    monkeypatch.setattr(
        pipeline, "tag_levels",
        lambda *_: [{"level": "L1", "status": "excluded", "voxels": 10000}])

    r = process_case(volume, seg, levels=["L1"])["results"]["L1"]
    assert r.editable and r.stats
    assert not r.included and r.accepted is None
    assert "instrumented level" in r.qc["exclusion_reason"]


def test_qc_fail_defaults_excluded_but_can_be_explicitly_included():
    volume, seg = _volume_and_seg()
    volume.hu[seg > 0] = 3000.0

    r = process_case(volume, seg, levels=["L1"])["results"]["L1"]
    assert r.qc["qc_status"] == "fail"
    assert r.editable and not r.included
    r.accepted = True
    assert r.included


def test_validation_counts_only_effectively_included_results():
    from spine_hu_tool.app.validate import build_report

    volume, seg = _volume_and_seg()
    volume.hu[seg > 0] = 3000.0
    case = process_case(volume, seg, levels=["L1"])

    report = build_report(case)
    assert report["n_measured"] == 0
    assert report["n_excluded"] == 1
    assert report["levels"]["L1"]["included"] is False
    assert "median_HU" not in report["levels"]["L1"]

    case["results"]["L1"].accepted = True
    included_report = build_report(case)
    assert included_report["n_measured"] == 1
    assert included_report["levels"]["L1"]["auto_excluded"] is True
    assert included_report["levels"]["L1"]["screening_override_reason"]


def test_level_without_measurement_geometry_stays_hard_excluded(monkeypatch):
    import spine_hu_tool.pipeline as pipeline

    seg = np.zeros((20, 20, 20), dtype=np.int16)
    seg[1:3, 1:6, 1:6] = label_id_for("L1")  # fewer than 100 voxels
    volume = Volume(
        hu=np.zeros(seg.shape, dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    monkeypatch.setattr(pipeline, "check_segmentation", lambda *_: _seg_check(False))
    monkeypatch.setattr(
        pipeline, "tag_levels",
        lambda *_: [{"level": "L1", "status": "clean"}])

    r = process_case(volume, seg, levels=["L1"])["results"]["L1"]
    assert r.qc["qc_status"] == "excluded"
    assert not r.editable and not r.included
    assert not r.stats


def test_invalid_level_measurement_error_does_not_abort_case(monkeypatch):
    import spine_hu_tool.pipeline as pipeline

    volume, seg = _volume_and_seg()
    monkeypatch.setattr(pipeline, "check_segmentation", lambda *_: _seg_check(False))
    monkeypatch.setattr(
        pipeline, "tag_levels",
        lambda *_: [{"level": "L1", "status": "clean"}])
    monkeypatch.setattr(
        pipeline, "measure_level",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad geometry")))

    r = process_case(volume, seg, levels=["L1"])["results"]["L1"]
    assert r.qc["qc_status"] == "excluded"
    assert "measurement unavailable: bad geometry" in r.qc["exclusion_reason"]


def test_valid_level_measurement_error_still_fails_the_case(monkeypatch):
    import spine_hu_tool.pipeline as pipeline

    volume, seg = _volume_and_seg()
    monkeypatch.setattr(pipeline, "check_segmentation", lambda *_: _seg_check(True))
    monkeypatch.setattr(
        pipeline, "tag_levels",
        lambda *_: [{"level": "L1", "status": "clean"}])
    monkeypatch.setattr(
        pipeline, "measure_level",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("system bug")))

    with pytest.raises(RuntimeError, match="system bug"):
        process_case(volume, seg, levels=["L1"])
