import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets
from spine_hu_tool.core import Volume
from spine_hu_tool.app.review_state import ReviewState
from spine_hu_tool.segmentation.totalseg_runner import label_id_for
from .synthetic import make_vertebra_phantom

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _state(audit_path=None):
    full, body, hu = make_vertebra_phantom()
    seg = np.where(full, label_id_for("L1"), 0).astype(np.int16)
    return ReviewState.from_volume(Volume(hu=hu, spacing=(1.0, 1.0, 1.0)),
                                   seg, levels=["L1"], audit_path=audit_path)


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


def test_chooser_collapses_to_one_row_per_study(test_data_dir):
    # opening a parent folder shows ONE selectable row per study (kernel
    # duplicates like STANDARD+BONE collapsed), and preselects a real series.
    # Asserted as an invariant so adding more studies/patients doesn't break it.
    from spine_hu_tool.app.viewer import MainWindow
    from spine_hu_tool.io.series_selector import select_ct_series
    win = MainWindow()
    best = win._load_folder(test_data_dir)
    assert best is not None

    _b, cands = select_ct_series(test_data_dir)
    axial = [c for c in cands if c.is_axial_ct]
    n_studies = len({c.study_uid for c in axial})
    # one selectable row per distinct study (kernel duplicates collapsed)
    assert len(win._row_series) == n_studies
    assert len({s.study_uid for s in win._row_series.values()}) == n_studies
    # preselected row resolves to the best axial CT
    sel = win._row_series.get(win.series_combo.currentIndex())
    assert sel is not None and sel.series_uid == best.series_uid
    # every *enabled* (non-header) combo row maps to a selectable SeriesInfo
    model = win.series_combo.model()
    for i in range(win.series_combo.count()):
        if model.item(i).isEnabled():
            assert i in win._row_series
