"""Regression tests on the newly-added studies (Anon1/Anon2).

These lock in how the pipeline handles two real-world situations the new data
exposed:

* instrumented (Anon2): a lumbar scan with spinal *hardware*. The instrumented
  levels must default to excluded while retaining an editable measurement, and
  the clean interior levels must still yield physiologic HU.
* longspine (Anon1): a long C7->sacrum scan whose end levels are clipped by the
  field of view. Interior levels must measure with a sane cranio-caudal HU
  gradient while FOV-clipped ends are excluded, not passed.

Both skip automatically when the cached NIfTI fixtures are absent (see
``work/cache_anon_fixtures.py``), matching the existing integration-test policy.
The full ``process_case`` is run once per study (module-scoped) because these
are full-resolution volumes.
"""
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _process(vol_name, seg_name, metadata):
    vol_p = os.path.join(ROOT, "work", vol_name)
    seg_p = os.path.join(ROOT, "work", seg_name)
    if not (os.path.exists(vol_p) and os.path.exists(seg_p)):
        pytest.skip(f"cached NIfTI not available ({vol_name}); "
                    "run work/cache_anon_fixtures.py after a pipeline run")
    from spine_hu_tool.io import load_volume_from_nifti
    from spine_hu_tool.segmentation import load_segmentation
    from spine_hu_tool.pipeline import process_case
    vol = load_volume_from_nifti(vol_p)
    vol.metadata = {**(vol.metadata or {}), **metadata}
    return process_case(vol, load_segmentation(seg_p))


@pytest.fixture(scope="module")
def instrumented_case():
    return _process("anon_instrumented.nii.gz", "anon_instrumented_seg.nii.gz",
                    {"kvp": 120.0, "manufacturer_model": "BrightSpeed"})


@pytest.fixture(scope="module")
def longspine_case():
    return _process("anon_longspine.nii.gz", "anon_longspine_seg.nii.gz",
                    {"kvp": 120.0, "manufacturer_model": "Revolution CT"})


def _measured(case):
    return {l: r for l, r in case["results"].items()
            if r.included and r.stats}


def test_instrumented_levels_are_quarantined(instrumented_case):
    res = instrumented_case["results"]
    # Hardware levels retain a tunable ROI but are excluded from reporting until
    # a reviewer explicitly includes them.
    for lvl in ("L3", "L5", "S1"):
        assert not res[lvl].included
        assert res[lvl].accepted is None
        assert res[lvl].editable
        assert "median_HU" in res[lvl].stats
    assert any("metal" in w or "instrumented" in w
               for w in res["L3"].qc.get("warnings", []))
    # a neighbor of an instrumented level is downgraded to review (streak/bloom)
    assert res["L2"].qc["qc_status"] == "review"
    # the clean interior thoraco-lumbar levels still measure in a physiologic band
    measured = _measured(instrumented_case)
    assert {"T10", "T11", "T12", "L1"} <= set(measured)
    for r in measured.values():
        assert 0 < r.stats["median_HU"] < 400


def test_instrumented_seg_flagged_and_calibration_notes(instrumented_case):
    # over-/under-segmented levels (speck T8, ballooned sacrum) -> suspect, with a
    # per-level (not global) reason; unknown GE scanner -> transparent calibration.
    case = instrumented_case
    assert case["seg_status"] in ("suspect", "invalid")
    check = case["seg_check"]
    bad = {lvl for lvl, v in check["levels"].items() if not v["valid"]}
    assert bad and all(check["levels"][lvl]["reasons"] for lvl in bad)
    notes = " ".join(case["calibration"]["notes"]).lower()
    assert "brightspeed" in notes or "not in table" in notes


def test_longspine_gradient_and_clipped_ends(longspine_case):
    res = longspine_case["results"]
    measured = _measured(longspine_case)
    assert len(measured) >= 8
    # plausible mid-thoracic > thoraco-lumbar-junction HU gradient
    if "T5" in measured and "T12" in measured:
        assert measured["T5"].stats["median_HU"] > measured["T12"].stats["median_HU"]
    for r in measured.values():
        assert 0 < r.stats["median_HU"] < 400
    # The ballooned/overlapping sacrum can retain a technically measurable ROI,
    # but segmentation screening keeps it out of reported results by default.
    assert not res["sacrum"].included
    assert res["sacrum"].qc.get("auto_excluded") or \
        res["sacrum"].qc["qc_status"] in ("excluded", "review", "fail")


def test_no_measured_level_reports_a_diagnosis(longspine_case):
    # the tool reports a reproducible measurement, never an osteoporosis class
    for r in longspine_case["results"].values():
        if r.stats:
            assert r.stats.get("classification") is None
