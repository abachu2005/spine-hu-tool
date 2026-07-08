import os
from spine_hu_tool.io.series_selector import (select_ct_series, group_series,
                                              list_studies)


def test_selects_standard_axial_lumbar(test_data_dir):
    best, cands = select_ct_series(os.path.join(test_data_dir, "10000680"))
    assert best is not None
    assert "STANDARD" in best.kernel.upper()
    assert best.n_files > 50
    # scouts/reformats/captures are rejected
    assert any(not c.is_axial_ct for c in cands)


def test_selects_standard_axial_thoracic(test_data_dir):
    best, cands = select_ct_series(os.path.join(test_data_dir, "100007EC"))
    assert best is not None
    assert "STANDARD" in best.kernel.upper()
    assert best.n_files > 100


def test_parent_folder_with_multiple_studies_is_grouped(test_data_dir):
    # the realistic physician upload: a parent folder holding many studies
    best, cands = select_ct_series(test_data_dir)
    assert best is not None
    axial = [c for c in cands if c.is_axial_ct]
    assert len(axial) >= 2

    # every axial series carries patient + study identifiers so duplicate
    # descriptions across patients are distinguishable in the chooser
    for c in axial:
        assert c.patient_id
        assert c.group_label and c.study_label
        assert "sl" in c.label

    groups = group_series(cands, axial_only=True)
    assert sum(len(g) for _label, g in groups) == len(axial)
    # the group holding the best series is listed first, best-first within it
    assert groups[0][1][0].series_uid == best.series_uid


def test_list_studies_collapses_kernel_duplicates(test_data_dir):
    # list_studies must collapse each STUDY to ONE best series (dropping kernel
    # duplicates like STANDARD+BONE), no matter how many studies/patients the
    # parent folder holds. Asserting the invariant -- one row per distinct study,
    # each the best-scored recon for that study -- keeps this robust as more test
    # data is dropped into the folder.
    best, cands = select_ct_series(test_data_dir)
    axial = [c for c in cands if c.is_axial_ct]
    n_studies = len({c.study_uid for c in axial})
    assert n_studies >= 2

    patients = list_studies(cands, axial_only=True)
    rows = [s for _p, studies in patients for s in studies]
    assert len(rows) == n_studies                       # exactly one row per study
    assert len({s.study_uid for s in rows}) == n_studies

    by_study: dict[str, list] = {}
    for c in axial:
        by_study.setdefault(c.study_uid, []).append(c)
    for s in rows:
        best_for_study = max(by_study[s.study_uid], key=lambda x: x.score)
        assert s.series_uid == best_for_study.series_uid   # kernel dup dropped
    # the global best is among the preselected study rows
    assert any(s.series_uid == best.series_uid for s in rows)


def test_selects_standard_axial_for_new_anon_studies(test_data_dir):
    # The newly-added Anon studies must auto-select a real STANDARD axial CT and
    # reject their reformatted/secondary recons (Anon2/Anon3 ship both).
    subs = [s for s in ("Anon1", "Anon2", "Anon3")
            if os.path.isdir(os.path.join(test_data_dir, s))]
    if not subs:
        import pytest
        pytest.skip("new Anon studies not present")
    for sub in subs:
        best, cands = select_ct_series(os.path.join(test_data_dir, sub))
        assert best is not None and best.is_axial_ct
        assert "STANDARD" in best.kernel.upper()
        assert "ORIGINAL" in best.image_type and "PRIMARY" in best.image_type
        assert best.reject_reason is None
        # any reformatted/secondary recon present must be rejected, not chosen
        for c in cands:
            if "REFORMATTED" in c.image_type or "SECONDARY" in c.image_type:
                assert not c.is_axial_ct


def test_group_series_falls_back_when_no_axial(test_data_dir):
    # scouts-only / odd folder should still surface something to pick
    _best, cands = select_ct_series(os.path.join(test_data_dir, "100007EC"))
    nonaxial = [c for c in cands if not c.is_axial_ct]
    groups = group_series(nonaxial, axial_only=True)
    assert sum(len(g) for _l, g in groups) == len(nonaxial)
