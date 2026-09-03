"""Scout body-habitus checks against the real studies.

The headline check is scan-rescan agreement: ``Test Data/10000680`` (lumbar) and
``Test Data/100007EC`` (thoracic) are the same patient scanned twice, on
different days with different table heights and different scan coverage. Their
levels overlap around T9-L1, so the habitus measured at those levels from two
independent scout pairs must agree -- that is the whole reproducibility claim,
and it is what would break first if a threshold, the couch rejection, or the
magnification correction regressed.

Skips automatically when the study folders or the cached segmentations are
absent, matching the existing integration-test policy.
"""
import os
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STUDIES = {
    "lumbar": ("Test Data/10000680", "work/lumbar_seg.nii"),
    "thoracic": ("Test Data/100007EC", "work/thoracic_seg.nii"),
}


def _habitus(study):
    folder, seg_rel = STUDIES[study]
    folder = os.path.join(ROOT, folder)
    seg_path = os.path.join(ROOT, seg_rel)
    if not os.path.isdir(folder) or not os.path.exists(seg_path):
        pytest.skip(f"{study} study or its cached segmentation is not available")
    from spine_hu_tool.io.series_selector import select_ct_series
    from spine_hu_tool.io.dicom_loader import load_series
    from spine_hu_tool.segmentation import load_segmentation
    from spine_hu_tool.scout import find_scouts, measure_habitus, level_z_ranges
    best, cands = select_ct_series(folder)
    volume = load_series(best.files)
    seg = load_segmentation(seg_path)
    if seg.shape != volume.shape:
        pytest.skip(f"cached {study} segmentation does not match the series")
    scouts = find_scouts(folder, study_uid=best.study_uid, candidates=cands)
    return measure_habitus(scouts, level_z_ranges(seg, volume)), scouts


@pytest.fixture(scope="module")
def lumbar():
    return _habitus("lumbar")


@pytest.fixture(scope="module")
def thoracic():
    return _habitus("thoracic")


def test_both_views_are_found_and_classified(lumbar):
    _out, scouts = lumbar
    assert {s.view for s in scouts} == {"AP", "LAT"}
    ap = next(s for s in scouts if s.view == "AP")
    lat = next(s for s in scouts if s.view == "LAT")
    # An AP projection samples x along its columns and images along y; the
    # lateral is the transpose of that. Getting this backwards would silently
    # swap width and depth.
    assert (ap.axis, ap.beam_axis) == (0, 1)
    assert (lat.axis, lat.beam_axis) == (1, 0)
    assert ap.sod_mm and ap.sid_mm and ap.sid_mm > ap.sod_mm
    # the scout must span the levels being measured
    assert ap.z_range[0] < -100 and ap.z_range[1] > 100


def test_habitus_is_reported_for_every_level_in_plausible_ranges(lumbar):
    out, _ = lumbar
    assert out["available"]
    levels = out["levels"]
    assert {"T12", "L1", "L2", "L3"} <= set(levels)
    for lvl, v in levels.items():
        # A real adult torso: wider than deep, both well inside the scan field.
        assert 200.0 < v["body_width_lr_mm"] < 500.0, lvl
        assert 150.0 < v["body_depth_ap_mm"] < 450.0, lvl
        assert v["body_width_lr_mm"] > v["body_depth_ap_mm"], lvl
        assert v["scout_warnings"] == [], lvl


def test_habitus_varies_smoothly_down_the_spine(lumbar):
    # Body size changes gradually between adjacent vertebrae; a jump would mean
    # the outline jumped onto the couch or off the patient at one level.
    out, _ = lumbar
    order = ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
    present = [l for l in order if l in out["levels"]]
    for key in ("body_width_lr_mm", "body_depth_ap_mm"):
        vals = [out["levels"][l][key] for l in present]
        assert max(abs(np.diff(vals))) < 25.0, key


def test_couch_is_excluded_from_the_lateral_depth(lumbar):
    # The couch sits ~50 mm posterior to the patient's back and is present on
    # every row. If it were included, the depth would exceed the width.
    out, scouts = lumbar
    lat = next(s for s in scouts if s.view == "LAT")
    from spine_hu_tool.scout import body_profile
    prof = body_profile(lat)
    # Posterior boundary must stay clear of the couch, which the scanner records
    # via TableHeight; the detected back surface is anterior to the table top.
    posterior = np.nanmedian(prof["hi_mm"])
    assert posterior < lat.table_height_mm + 30.0
    assert np.nanmedian(prof["width_mm"]) < 350.0


def test_scan_rescan_agreement_between_the_two_studies(lumbar, thoracic):
    lum, _ = lumbar
    tho, _ = thoracic
    common = sorted(set(lum["levels"]) & set(tho["levels"]))
    assert len(common) >= 4, f"expected overlapping levels, got {common}"
    for key, tol_bias, tol_mean in (("body_width_lr_mm", 6.0, 8.0),
                                    ("body_depth_ap_mm", 6.0, 8.0),
                                    ("body_effective_diameter_mm", 5.0, 6.0)):
        d = np.array([tho["levels"][l][key] - lum["levels"][l][key]
                      for l in common])
        assert abs(d.mean()) < tol_bias, f"{key} bias {d.mean():.1f} mm over {common}"
        assert np.abs(d).mean() < tol_mean, f"{key} mean|delta| {np.abs(d).mean():.1f} mm"


def test_magnification_correction_is_small_but_present(lumbar):
    out, _ = lumbar
    applied = 0
    for v in out["levels"].values():
        for key, val, raw in (
                ("scout_ap_magnification", "body_width_lr_mm",
                 "body_width_lr_uncorrected_mm"),
                ("scout_lat_magnification", "body_depth_ap_mm",
                 "body_depth_ap_uncorrected_mm")):
            f = v[key]
            # A correctly centered patient needs only a few percent; anything
            # larger means the geometry or the center estimate is wrong.
            assert 0.95 < f < 1.05
            assert v[val] == pytest.approx(v[raw] * f, abs=0.15)
            applied += abs(f - 1.0) > 0.002
    # ...and it is not a no-op: most levels are corrected by a visible amount.
    assert applied >= len(out["levels"])


def test_study_without_scouts_degrades_gracefully():
    # The de-identified re-exports dropped their localizers. The tool must say so
    # and still measure HU, never fail.
    folder = os.path.join(ROOT, "Test Data", "Anon2")
    if not os.path.isdir(folder):
        pytest.skip("Anon2 study not available")
    from spine_hu_tool.app.analysis import has_scouts
    from spine_hu_tool.scout import find_scouts, measure_habitus
    assert has_scouts(folder) is False
    assert find_scouts(folder) == []
    out = measure_habitus([], {"L1": (0.0, 10.0)})
    assert out["available"] is False and out["levels"] == {}


def test_scout_overlays_are_written_on_export(lumbar, tmp_path):
    out, _ = lumbar
    from spine_hu_tool.export.writers import write_scout_overlays
    case = {"scout": out}
    paths = write_scout_overlays(case, str(tmp_path))
    assert len(paths) == 2
    for p in paths:
        assert os.path.getsize(p) > 1000
