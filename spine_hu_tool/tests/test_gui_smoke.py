import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets
from spine_hu_tool.core import Volume
from spine_hu_tool.app.review_state import ReviewState
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from spine_hu_tool.io.series_selector import SeriesInfo
from spine_hu_tool.io.study_discovery import studies_from_candidates
from .synthetic import make_vertebra_phantom

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _state(audit_path=None):
    full, body, hu = make_vertebra_phantom()
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    return ReviewState.from_volume(Volume(hu=hu, spacing=(1.0, 1.0, 1.0)),
                                   seg, levels=["L1"], audit_path=audit_path)


def _fake_studies(n):
    cands = []
    for i in range(n):
        s = SeriesInfo(
            series_uid=f"u{i}", study_uid=f"st{i}", modality="CT", series_number=1,
            description="series", image_type=("ORIGINAL", "PRIMARY", "AXIAL"),
            kernel="STANDARD", n_files=10, files=[f"/data/{i}/a.dcm"],
            pixel_spacing=(1.0, 1.0), slice_thickness=1.0, rows=512, cols=512,
            rescale_slope=1.0, rescale_intercept=-1024.0, uniform_spacing=True,
            z_spacing=1.0, patient_id=f"P{i}", study_description=f"CT study {i}",
            study_date="20240101")
        s.is_axial_ct = True
        s.score = 100 - i
        cands.append(s)
    return studies_from_candidates(cands)


def test_mainwindow_loads_and_renders():
    from spine_hu_tool.app.viewer import MainWindow
    win = MainWindow()
    win.load_state(_state())
    assert win.stack.currentIndex() == 1
    assert win.level_list.count() == 1
    win.level_list.setCurrentRow(0)
    win.render()                       # exercises tri-planar rendering
    # edit through the GUI -> backend
    win._on_radius(40)
    assert abs(win.state.results["L1"].radius_mm - 4.0) < 1e-6
    win._decide(True)
    assert win.state.results["L1"].accepted is True


def test_cursor_drag_moves_roi_and_records_once(tmp_path):
    # Dragging the ROI emits many provisional moves (record=False) then one final
    # move on release (record=True): the center tracks the cursor live, but only
    # the final position is written to the audit log.
    from spine_hu_tool.app.viewer import MainWindow
    win = MainWindow()
    win.load_state(_state(audit_path=str(tmp_path / "audit.json")))
    win.level_list.setCurrentRow(0)
    win.render()
    start = win.state.results["L1"].center_idx
    n_before = len(win.state.audit.entries)

    handler = win._make_click_handler(win.ax)   # axial plane: cols=X, rows=Y
    sh, sv = win.ax.sh, win.ax.sv
    tx, ty = start[0] + 4, start[1] + 3
    for k in range(1, 4):                        # provisional drag steps
        handler((start[0] + k) * sh, (start[1] + k) * sv, False)
    handler(tx * sh, ty * sv, True)              # release -> record

    assert win.state.results["L1"].center_idx[:2] == (tx, ty)   # tracked cursor
    assert len(win.state.audit.entries) == n_before + 1         # one entry only


def test_undo_reverts_move_resize_and_decision():
    # Cmd+Z (-> _undo) reverts the last edit; a drag gesture is one undo step.
    from spine_hu_tool.app.viewer import MainWindow
    win = MainWindow()
    win.load_state(_state())
    win.level_list.setCurrentRow(0)
    win.render()
    orig_center = win.state.results["L1"].center_idx
    orig_radius = win.state.results["L1"].radius_mm

    # 1) a multi-step drag -> single undo step
    handler = win._make_click_handler(win.ax)
    sh, sv = win.ax.sh, win.ax.sv
    tx, ty = orig_center[0] + 5, orig_center[1] + 2
    for k in range(1, 4):
        handler((orig_center[0] + k) * sh, (orig_center[1] + k) * sv, False)
    handler(tx * sh, ty * sv, True)
    assert win.state.results["L1"].center_idx[:2] == (tx, ty)

    # 2) a radius change (slider works in 0.1 mm ticks)
    new_val = int(round((orig_radius + 2.0) * 10))
    win._on_radius(new_val)
    assert abs(win.state.results["L1"].radius_mm - new_val / 10.0) < 1e-6

    # 3) a decision
    win._decide(True)
    assert win.state.results["L1"].accepted is True

    win._undo()                                  # undo decision
    assert win.state.results["L1"].accepted is not True
    win._undo()                                  # undo radius
    assert abs(win.state.results["L1"].radius_mm - orig_radius) < 1e-6
    win._undo()                                  # undo the whole drag at once
    assert win.state.results["L1"].center_idx == orig_center
    assert not win.state.can_undo()              # stack emptied


def test_enter_accepts_current_level():
    # pressing Enter (-> _accept_advance) marks the current level accepted.
    from spine_hu_tool.app.viewer import MainWindow
    win = MainWindow()
    win.load_state(_state())
    win.level_list.setCurrentRow(0)
    assert win.state.results["L1"].accepted is not True
    win._accept_advance()
    assert win.state.results["L1"].accepted is True


def test_excluded_level_cannot_be_accepted_or_rejected():
    # an excluded level has no measurement: accept/reject is disabled and a no-op,
    # and Enter skips past it without marking a decision.
    from spine_hu_tool.app.viewer import MainWindow
    win = MainWindow()
    win.load_state(_state())
    win.state.results["L1"].qc["qc_status"] = "excluded"   # simulate exclusion
    win._on_level_changed(0)
    assert not win.accept_btn.isEnabled()
    assert not win.reject_btn.isEnabled()
    win._decide(True)
    assert win.state.results["L1"].accepted is None        # decision not applied
    win._accept_advance()
    assert win.state.results["L1"].accepted is None        # Enter did not accept


def test_chooser_lists_one_row_per_study(test_data_dir):
    # opening a parent folder shows ONE checkable row per study (kernel
    # duplicates like STANDARD+BONE collapsed), best study checked by default.
    from spine_hu_tool.app.viewer import MainWindow
    from spine_hu_tool.io.series_selector import select_ct_series
    win = MainWindow()
    best = win._load_folder(test_data_dir)
    assert best is not None

    _b, cands = select_ct_series(test_data_dir)
    axial = [c for c in cands if c.is_axial_ct]
    n_studies = len({c.study_uid for c in axial})
    assert win.study_list.count() == n_studies
    assert len({s.study_uid for s in win._studies}) == n_studies
    # first (best) study checked by default; it resolves to the best axial CT
    assert win.study_list.item(0).checkState() == QtCore.Qt.Checked
    assert win._studies[0].best_series.series_uid == best.series_uid


def test_multiselect_and_select_all():
    from spine_hu_tool.app.viewer import MainWindow
    win = MainWindow()
    win.folder = "/data"
    win._studies = _fake_studies(3)
    # populate the list the way _load_folder does
    win.study_list.blockSignals(True)
    win.study_list.clear()
    for i in range(3):
        it = QtWidgets.QListWidgetItem(f"study {i}")
        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
        it.setData(QtCore.Qt.UserRole, i)
        it.setCheckState(QtCore.Qt.Checked if i == 0 else QtCore.Qt.Unchecked)
        win.study_list.addItem(it)
    win.study_list.blockSignals(False)

    assert len(win._checked_studies()) == 1
    win.select_all_cb.setChecked(True)
    assert len(win._checked_studies()) == 3
    win.select_all_cb.setChecked(False)
    assert len(win._checked_studies()) == 0


def test_batch_worker_saves_runs_and_grid(tmp_path, monkeypatch):
    # BatchWorker over several studies, with analyze_dataset STUBBED so no real
    # segmentation runs. Runs are saved to the library and the grid renders.
    monkeypatch.setenv("SPINE_HU_RUNS_DIR", str(tmp_path / "runs"))
    from spine_hu_tool.app import viewer as vmod
    from spine_hu_tool.app.viewer import MainWindow, BatchWorker

    def _fake_analyze(folder, series=None, **kw):
        return _state()
    monkeypatch.setattr(vmod, "analyze_dataset", _fake_analyze)

    studies = _fake_studies(3)
    worker = BatchWorker("/data", studies, "centroid_volume_sphere",
                         local=False, fast=False)
    worker.run()                          # run synchronously (no thread)
    from spine_hu_tool.app import run_store as rs
    batch = rs.load_batch(worker.batch_id)
    assert len(batch["run_ids"]) == 3
    assert all(rs.load_run(rid)["status"] == "ready" for rid in batch["run_ids"])

    win = MainWindow()
    win.show_batch(worker.batch_id)
    assert win.stack.currentIndex() == 2
    # one card per study
    assert win.results_vbox.count() == 3


def test_batch_records_failure_without_aborting(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINE_HU_RUNS_DIR", str(tmp_path / "runs"))
    from spine_hu_tool.app import viewer as vmod
    from spine_hu_tool.app.viewer import BatchWorker

    calls = {"n": 0}

    def _flaky(folder, series=None, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bad study")
        return _state()
    monkeypatch.setattr(vmod, "analyze_dataset", _flaky)

    worker = BatchWorker("/data", _fake_studies(3), "centroid_volume_sphere")
    worker.run()
    from spine_hu_tool.app import run_store as rs
    statuses = [rs.load_run(rid)["status"] for rid in rs.load_batch(worker.batch_id)["run_ids"]]
    assert statuses.count("failed") == 1 and statuses.count("ready") == 2


def test_open_run_and_finalize(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINE_HU_RUNS_DIR", str(tmp_path / "runs"))
    from spine_hu_tool.app.viewer import MainWindow
    from spine_hu_tool.app import run_store as rs

    state = _state()
    state.set_decision("L1", True)
    meta = {"series_uid": "1.2.3", "study_description": "CT", "patient_label": "P1",
            "study_date": "20240101", "source_folder": "/x", "files": ["/x/a.dcm"]}
    rec = rs.build_record(state, meta, backend="cloud", resolution="full",
                          mode="centroid_volume_sphere")
    rs.save_run(rec)

    win = MainWindow()
    win._grid_context = ("past", None)
    # reopen returns a fresh state (stub so no DICOM/seg needed)
    monkeypatch.setattr(rs, "reopen_run", lambda rid, apply_review=True: _state())
    win.open_run(rec["id"])
    assert win.stack.currentIndex() == 1
    assert win.current_run_id == rec["id"]
    assert not win.back_btn.isHidden()          # "Back to results" shown

    # finalize marks reviewed and writes an export dir in the library
    win.state.set_decision("L1", True)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    win.finalize()
    updated = rs.load_run(rec["id"])
    assert updated["status"] == "reviewed"
    assert updated["export_dir"] and os.path.isdir(updated["export_dir"])


def test_past_runs_lists_batches_and_singles(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINE_HU_RUNS_DIR", str(tmp_path / "runs"))
    from spine_hu_tool.app.viewer import MainWindow
    from spine_hu_tool.app import run_store as rs

    state = _state()
    meta = {"series_uid": "s", "study_description": "CT", "patient_label": "P",
            "source_folder": "/x", "files": ["/x/a.dcm"]}
    # one standalone single run
    single = rs.build_record(state, meta, backend="cloud", resolution="full",
                             mode="centroid_volume_sphere")
    rs.save_run(single)
    # one batch of two runs
    ids = []
    for _ in range(2):
        r = rs.build_record(state, meta, backend="cloud", resolution="full",
                            mode="centroid_volume_sphere")
        rs.save_run(r); ids.append(r["id"])
    b = rs.new_batch(ids, source_folder="/x"); rs.save_batch(b)

    win = MainWindow()
    win.open_past_runs()
    assert win.stack.currentIndex() == 2
    # 1 batch card + 1 standalone single = 2 top-level entries
    assert win.results_vbox.count() == 2
