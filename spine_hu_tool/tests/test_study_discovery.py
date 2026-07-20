"""Study discovery grouping tests (synthetic SeriesInfo -- no DICOM needed)."""
from spine_hu_tool.io.series_selector import SeriesInfo
from spine_hu_tool.io.study_discovery import studies_from_candidates


def _series(uid, study, pid, score, axial=True, kernel="STANDARD", files=None):
    s = SeriesInfo(
        series_uid=uid, study_uid=study, modality="CT", series_number=1,
        description="series", image_type=("ORIGINAL", "PRIMARY", "AXIAL"),
        kernel=kernel, n_files=100,
        files=files or [f"/data/{pid}/{study}/{uid}/img.dcm"],
        pixel_spacing=(1.0, 1.0), slice_thickness=1.0, rows=512, cols=512,
        rescale_slope=1.0, rescale_intercept=-1024.0, uniform_spacing=True,
        z_spacing=1.0, patient_id=pid, study_description=f"CT {study}",
        study_date="20240101")
    s.is_axial_ct = axial
    s.score = score
    return s


def test_one_record_per_study_ordered_best_first():
    cands = [
        _series("a", "stA", "P1", 120),
        _series("b", "stB", "P1", 110),
        _series("c", "stC", "P2", 130),
    ]
    recs = studies_from_candidates(cands)
    assert [r.study_uid for r in recs] == ["stC", "stA", "stB"]  # by best score
    assert all(r.n_series == 1 for r in recs)


def test_kernel_duplicates_collapse_to_best_but_keep_all_series():
    cands = [
        _series("std", "stA", "P1", 120, kernel="STANDARD"),
        _series("bone", "stA", "P1", 105, kernel="BONE"),
    ]
    recs = studies_from_candidates(cands)
    assert len(recs) == 1
    assert recs[0].best_series.series_uid == "std"     # best recon chosen
    assert recs[0].n_series == 2                        # both kept as siblings


def test_multi_patient_grouping():
    cands = [
        _series("a", "stA", "P1", 120),
        _series("b", "stB", "P2", 118),
        _series("c", "stC", "P3", 115),
    ]
    recs = studies_from_candidates(cands)
    assert len({r.patient_id for r in recs}) == 3
    assert len(recs) == 3


def test_axial_only_filters_but_falls_back_when_none():
    non_axial = [_series("x", "stX", "P1", -2, axial=False)]
    # axial_only with no axial -> still surface something rather than nothing
    recs = studies_from_candidates(non_axial, axial_only=True)
    assert len(recs) == 1
    assert recs[0].has_axial is False


def test_source_folder_is_common_ancestor():
    files = ["/root/P1/stA/s1/a.dcm", "/root/P1/stA/s1/b.dcm"]
    recs = studies_from_candidates([_series("s1", "stA", "P1", 120, files=files)])
    assert recs[0].source_folder == "/root/P1/stA/s1"


def test_as_meta_has_reopen_fields():
    files = ["/root/P1/stA/s1/a.dcm"]
    rec = studies_from_candidates([_series("s1", "stA", "P1", 120, files=files)])[0]
    meta = rec.as_meta()
    for key in ("series_uid", "study_uid", "patient_id", "source_folder", "files"):
        assert key in meta
    assert meta["files"] == files
