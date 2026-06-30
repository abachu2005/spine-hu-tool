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
    # the parent folder has 4 axial series = 2 studies x {STANDARD, BONE} kernels;
    # list_studies must collapse each study to ONE best (STANDARD) series.
    best, cands = select_ct_series(test_data_dir)
    patients = list_studies(cands, axial_only=True)
    rows = [s for _p, studies in patients for s in studies]
    assert len(rows) == 2                                   # one per study, not 4
    assert len({s.study_uid for s in rows}) == 2            # the two real studies
    for s in rows:
        assert "STANDARD" in s.kernel.upper()               # bone duplicates dropped
    # single patient in this dataset
    assert len(patients) == 1
    # the global best is among (and is) the preselected study series
    assert any(s.series_uid == best.series_uid for s in rows)


def test_group_series_falls_back_when_no_axial(test_data_dir):
    # scouts-only / odd folder should still surface something to pick
    _best, cands = select_ct_series(os.path.join(test_data_dir, "100007EC"))
    nonaxial = [c for c in cands if not c.is_axial_ct]
    groups = group_series(nonaxial, axial_only=True)
    assert sum(len(g) for _l, g in groups) == len(nonaxial)
