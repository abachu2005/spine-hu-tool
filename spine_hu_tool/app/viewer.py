"""Greyscale, PACS-style physician review desktop app (PySide6 + pyqtgraph).

Workflow: open DICOM folder -> auto series-select -> Analyze (background) ->
tri-planar review (accept / reject / adjust radius / move center) -> export.

The GUI is a thin shell over `ReviewState` (all edits route through the
deterministic backend). Rendering is greyscale CT with low-saturation
translucent overlays and a one-click toggle back to clean greyscale.
"""
from __future__ import annotations
import os
import numpy as np

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .review_state import ReviewState
from .analysis import analyze_dataset
from . import applog
from . import run_store
from ..errors import UserFacingError
from ..roi.modes import DEFAULT_MODE, EXPOSED_MODES, MODE_LABELS
from ..segmentation.backends import resolve_seg_url
from ..segmentation.totalseg_runner import local_seg_available

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "#1a1a1d")
pg.setConfigOption("foreground", "#d8d8d8")

# overlay colors (low-saturation, PACS-like)
COL_BODY = (120, 200, 140)
COL_INNER = (110, 170, 220)
COL_ROI = (240, 210, 90)

STATUS_COLOR = {"pass": "#5fbf6f", "review": "#d8a23a", "fail": "#c5544a",
                "excluded": "#c5544a", None: "#888888"}

DARK_QSS = """
QWidget { background:#1a1a1d; color:#d8d8d8; font-size:13px; }
QPushButton { background:#2c2c31; border:1px solid #3a3a40; padding:7px 12px;
              border-radius:4px; }
QPushButton:hover { background:#36363c; }
QPushButton:disabled { color:#666; }
QListWidget { background:#202024; border:1px solid #33333a; }
QListWidget::item:selected { background:#3a4a5a; }
QLabel#title { font-size:20px; font-weight:600; }
QGroupBox { border:1px solid #33333a; border-radius:5px; margin-top:8px; }
QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }
"""


def window_to_rgb(hu2d, overlays, wc, ww):
    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    g = np.clip((hu2d - lo) / max(1e-3, (hi - lo)), 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    for mask, color, alpha in overlays:
        if mask is None or not mask.any():
            continue
        c = np.asarray(color, float) / 255.0
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * c
    return (rgb * 255).astype(np.uint8)


class PlaneView(pg.GraphicsLayoutWidget):
    """One orthogonal plane with image, crosshair, and orientation labels."""
    # image col, row (physical mm), and whether to record the edit (audit). A
    # drag emits many provisional moves (record=False) then a final one on
    # release (record=True); a single click emits one (record=True).
    moved = QtCore.Signal(float, float, bool)

    def __init__(self, name):
        super().__init__()
        # Render reformats with sharp voxel blocks, not smooth interpolation: on
        # thick-slice CT the through-plane axis is stretched ~5x, and smoothing
        # blurs the true ROI->endplate gap into an apparent contact.
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.vb = self.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.img.setAutoDownsample(True)
        self.vb.addItem(self.img)
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#e0c050", width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen("#e0c050", width=1))
        self.vb.addItem(self.vline); self.vb.addItem(self.hline)
        self.title = pg.TextItem(name, color="#9fb6c9")
        self.vb.addItem(self.title)
        self._labels = []
        self.sh = self.sv = 1.0
        self.img.mouseClickEvent = self._on_click
        self.img.mouseDragEvent = self._on_drag
        self.setCursor(QtCore.Qt.CrossCursor)

    def _on_click(self, ev):
        ev.accept()
        p = self.img.mapToView(ev.pos())
        self.moved.emit(float(p.x()), float(p.y()), True)

    def _on_drag(self, ev):
        # Left-drag moves the ROI; other buttons fall through to the view (zoom).
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        p = self.img.mapToView(ev.pos())
        self.moved.emit(float(p.x()), float(p.y()), bool(ev.isFinish()))

    def set_orientation_labels(self, left, right, top, bottom, w, h):
        for t in self._labels:
            self.vb.removeItem(t)
        self._labels = []
        for txt, x, y in [(left, 0, h / 2), (right, w, h / 2),
                          (top, w / 2, 0), (bottom, w / 2, h)]:
            ti = pg.TextItem(txt, color="#7fa0b5"); ti.setPos(x, y)
            self.vb.addItem(ti); self._labels.append(ti)

    def set_data(self, hu2d, overlays, sh, sv, cross_hv=None,
                 labels=None, reset_range=True):
        self.sh, self.sv = sh, sv
        rgb = window_to_rgb(hu2d, overlays, *self._wl) if hasattr(self, "_wl") \
            else window_to_rgb(hu2d, overlays, 400, 1500)
        self.img.setImage(rgb)
        tr = QtGui.QTransform()
        tr.scale(sh, sv)
        self.img.setTransform(tr)
        h, w = hu2d.shape[0] * sv, hu2d.shape[1] * sh
        self.title.setPos(4 * sh, 4 * sv)
        if cross_hv is not None:
            col, row = cross_hv
            self.vline.setPos(col * sh); self.hline.setPos(row * sv)
        if labels:
            self.set_orientation_labels(*labels, w, h)
        # Only re-fit the view when new content is loaded; during a live ROI drag
        # (reset_range=False) keep the user's current zoom/pan instead of jumping.
        if reset_range:
            self.vb.autoRange(padding=0.02)


class AnalyzeWorker(QtCore.QThread):
    progress = QtCore.Signal(str, float)
    finished_state = QtCore.Signal(object)
    failed = QtCore.Signal(object)          # carries the exception object

    def __init__(self, folder, series, mode, reviewer, seg_url=None, api_key=None,
                 local=False, fast=None, scout=True):
        super().__init__()
        self.folder, self.series, self.mode, self.reviewer = folder, series, mode, reviewer
        self.seg_url, self.api_key, self.local = seg_url, api_key, local
        self.fast = fast
        self.scout = scout

    def run(self):
        try:
            st = analyze_dataset(self.folder, series=self.series, mode=self.mode,
                                 only_clean=True, reviewer=self.reviewer,
                                 seg_url=self.seg_url, api_key=self.api_key,
                                 local=self.local, fast=self.fast, scout=self.scout,
                                 progress=lambda m, f: self.progress.emit(m, f))
            self.finished_state.emit(st)
        except Exception as e:  # surfaced to the user
            applog.log_exception("analysis failed", e)
            self.failed.emit(e)


class BatchWorker(QtCore.QThread):
    """Analyze several studies sequentially, saving each into the run library.

    Runs off the UI thread. Each study is analyzed independently so one failure
    (e.g. an unreadable study) does not abort the batch -- it is recorded as a
    failed run and the batch continues. Emits per-study completion and a final
    batch id.
    """
    progress = QtCore.Signal(str, float)
    study_done = QtCore.Signal(str, str, str)     # run_id, label, status
    finished_batch = QtCore.Signal(str)           # batch_id
    failed = QtCore.Signal(str)

    def __init__(self, folder, studies, mode, reviewer="physician",
                 seg_url=None, local=False, fast=None, scout=True):
        super().__init__()
        self.folder, self.studies, self.mode, self.reviewer = folder, studies, mode, reviewer
        self.seg_url, self.local, self.fast = seg_url, local, fast
        self.scout = scout
        self.batch_id = run_store.new_batch_id()
        self._backend = "local" if local else "cloud"
        self._resolution = "fast" if fast else "full"

    def run(self):
        try:
            run_ids = []
            n = len(self.studies)
            for i, study in enumerate(self.studies):
                label = study.best_series.study_description or study.patient_label
                meta = study.as_meta()
                self.progress.emit(f"Analyzing study {i + 1} of {n}: {label}", i / max(1, n))
                try:
                    state = analyze_dataset(
                        self.folder, series=study.best_series, mode=self.mode,
                        only_clean=True, reviewer=self.reviewer,
                        seg_url=self.seg_url, local=self.local, fast=self.fast,
                        scout=self.scout)
                    rec = run_store.build_record(
                        state, meta, backend=self._backend,
                        resolution=self._resolution, mode=self.mode,
                        reviewer=self.reviewer, batch_id=self.batch_id,
                        status="ready")
                    run_store.save_run(rec)
                    run_ids.append(rec["id"])
                    self.study_done.emit(rec["id"], label, "ready")
                except Exception as e:  # one bad study must not kill the batch
                    rec = run_store.failed_record(
                        meta, str(e), backend=self._backend,
                        resolution=self._resolution, mode=self.mode,
                        batch_id=self.batch_id)
                    run_store.save_run(rec)
                    run_ids.append(rec["id"])
                    self.study_done.emit(rec["id"], label, "failed")
            batch = run_store.new_batch(run_ids, source_folder=self.folder)
            batch["id"] = self.batch_id
            run_store.save_batch(batch)
            self.progress.emit("Batch complete.", 1.0)
            self.finished_batch.emit(self.batch_id)
        except Exception as e:  # unexpected: surface it
            applog.log_exception("batch analysis failed", e)
            self.failed.emit(e)


class SetupWorker(QtCore.QThread):
    """Installs (or repairs) the local-segmentation runtime (torch +
    TotalSegmentator + weights) off the UI thread, streaming progress lines."""
    progress = QtCore.Signal(str)
    done = QtCore.Signal()
    failed = QtCore.Signal(object)          # carries the exception object

    def __init__(self, repair: bool = False):
        super().__init__()
        self.repair = repair

    def run(self):
        from ..segmentation.local_setup import (
            classify_setup_error, repair_local_seg, setup_local_seg)
        try:
            fn = repair_local_seg if self.repair else setup_local_seg
            fn(progress=lambda m: self.progress.emit(m))
            self.done.emit()
        except Exception as e:  # surfaced to the user
            applog.log_exception(
                "local-seg repair failed" if self.repair else "local-seg setup failed", e)
            self.failed.emit(classify_setup_error(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spine Vertebral-HU Tool")
        self.resize(1500, 950)
        self.setStyleSheet(DARK_QSS)
        self.state: ReviewState | None = None
        self.folder = None
        self.series = None
        self.current_level = None
        self._studies = []            # StudyRecord list for the opened folder
        self.current_run_id = None    # run library id of the run under review
        self._grid_context = None     # None | ("past", None) | ("batch", batch_id)
        # display state
        self.wc, self.ww = 400.0, 1500.0
        self.show_body = True
        self.show_inner = True
        self.show_roi = True
        self._drag_active = False     # coalesce a cursor drag into one undo step
        self._radius_active = False   # coalesce a slider sweep into one undo step
        self._populating = False      # suppress combo signals while repopulating
        self._candidates = None
        self._build()
        # First launch: show the pilot / non-PHI disclaimer once the window is up.
        QtCore.QTimer.singleShot(0, self._maybe_show_first_run_disclaimer)

    # ---- first-run pilot / PHI disclaimer --------------------------------
    def _maybe_show_first_run_disclaimer(self):
        settings = QtCore.QSettings("SpineHUTool", "SpineHUTool")
        if settings.value("acceptedPilotDisclaimer", False, type=bool):
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Pilot use — please read")
        box.setText("Spine HU Tool — research / pilot software")
        box.setInformativeText(
            "This tool is for pilot and research use only and is NOT a "
            "validated diagnostic device.\n\n"
            "By default, segmentation runs on a shared cloud service: the "
            "scans you open are uploaded to that hosted service for "
            "processing. Do NOT use protected health information (PHI) — "
            "use de-identified data only.\n\n"
            "All runs — including runs where segmentation is computed on "
            "this computer — are archived to the hosted service (inputs, "
            "outputs, and logs) for quality assurance, record keeping, and "
            "troubleshooting.\n\n"
            "By continuing you confirm you understand and accept these terms.")
        agree = box.addButton("I understand and accept", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Quit", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is agree:
            settings.setValue("acceptedPilotDisclaimer", True)
        else:
            self.close()

    # ---- UI construction -------------------------------------------------
    def _build(self):
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._landing_page())   # 0
        self.stack.addWidget(self._review_page())     # 1
        self.stack.addWidget(self._results_page())    # 2 (batch complete / past runs)
        # Cmd+Z / Ctrl+Z reverts the last ROI edit (move, resize, recompute,
        # accept/reject); each gesture is a single undo step.
        self._undo_sc = QtGui.QShortcut(QtGui.QKeySequence.Undo, self)
        self._undo_sc.activated.connect(self._undo)
        # Enter / Return accepts the current level and advances to the next, for
        # fast keyboard-driven review.
        self._accept_scs = []
        for _key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            sc = QtGui.QShortcut(QtGui.QKeySequence(_key), self)
            sc.activated.connect(self._accept_advance)
            self._accept_scs.append(sc)

    def _landing_page(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addStretch()
        title = QtWidgets.QLabel("Spine Vertebral-HU Tool"); title.setObjectName("title")
        title.setAlignment(QtCore.Qt.AlignCenter)
        sub = QtWidgets.QLabel("Open a DICOM folder to measure trabecular HU per vertebra.")
        sub.setAlignment(QtCore.Qt.AlignCenter); sub.setStyleSheet("color:#9a9aa2;")
        btn = QtWidgets.QPushButton("Open DICOM folder...")
        btn.setFixedWidth(260); btn.clicked.connect(self.open_folder)
        self.past_runs_btn = QtWidgets.QPushButton("View past runs...")
        self.past_runs_btn.setFixedWidth(260)
        self.past_runs_btn.clicked.connect(self.open_past_runs)
        # The hosted endpoint remains available for deployments that provide
        # credentials, while full release builds prefer their bundled local
        # segmentation runtime by default.
        self.seg_url_edit = QtWidgets.QLineEdit(resolve_seg_url(None) or "")
        self.seg_url_edit.setFixedWidth(420)
        self.seg_url_edit.setPlaceholderText("Segmentation server URL (cloud default)")
        # Local segmentation toggle: when checked, the heavy TotalSegmentator step
        # runs on this machine instead of the cloud service (requires the optional
        # local-seg dependencies / TotalSegmentator to be installed). NOTE: this
        # is compute-location wording only -- every run (local or cloud) is
        # archived to the hosted service for QA/troubleshooting, so it must NOT
        # promise "no upload".
        self.local_seg_cb = QtWidgets.QCheckBox(
            "Run segmentation on this computer")
        self.local_seg_cb.setToolTip(
            "Compute the segmentation locally instead of on the cloud service. "
            "The offline build has everything it needs built in; otherwise use "
            "\u201cSet up local segmentation\u201d once. Full-res needs more "
            "RAM \u2014 pick Fast on a low-memory machine. Note: run data is "
            "still archived to the hosted service for quality assurance and "
            "troubleshooting.")
        self.local_seg_cb.toggled.connect(self._on_local_toggled)
        # Segmentation resolution: full-res (1.5 mm, matches cloud, best ROI
        # placement) vs fast (3 mm, much lighter on RAM). Full-res is the
        # default; a low-RAM machine is warned before a full-res local run.
        self.res_combo = QtWidgets.QComboBox(); self.res_combo.setFixedWidth(420)
        self.res_combo.addItem("Full resolution (1.5 mm, recommended)", False)
        self.res_combo.addItem("Fast (3 mm, low memory)", True)
        # One-time installer for the local ML runtime (torch + TotalSegmentator
        # + weights) into a user-managed environment outside the app bundle.
        self.setup_local_btn = QtWidgets.QPushButton("Set up local segmentation...")
        self.setup_local_btn.setFixedWidth(420)
        self.setup_local_btn.clicked.connect(self.setup_local_seg)
        self._refresh_local_seg_state(prefer_local=True)
        # Body habitus from the scout films. Off unless the opened study actually
        # ships localizers, which many de-identified exports drop.
        self.scout_cb = QtWidgets.QCheckBox("Record scout film measurements")
        self.scout_cb.setToolTip(
            "Measure the patient's body width and depth at each vertebral level "
            "from the scout (localizer) films. The axial images are cropped to "
            "the spine, so the body outline only exists on the scouts.")
        self.scout_cb.setChecked(True)
        self._set_scout_available(True)
        self.roi_mode_combo = QtWidgets.QComboBox(); self.roi_mode_combo.setFixedWidth(420)
        for label, key in EXPOSED_MODES.items():
            self.roi_mode_combo.addItem(label, key)
        self.roi_mode_combo.setCurrentText(MODE_LABELS.get(DEFAULT_MODE, "centroid"))
        # Multi-select study chooser: one checkable row per study so several
        # studies (or all of them) can be batch-processed in one go.
        self.select_all_cb = QtWidgets.QCheckBox("Select all studies")
        self.select_all_cb.setFixedWidth(420)
        self.select_all_cb.setVisible(False)
        self.select_all_cb.toggled.connect(self._on_select_all)
        self.study_list = QtWidgets.QListWidget(); self.study_list.setFixedWidth(420)
        self.study_list.setFixedHeight(150)
        self.study_list.setVisible(False)
        self.study_list.itemChanged.connect(self._on_study_item_changed)
        self.analyze_btn = QtWidgets.QPushButton("Analyze selected")
        self.analyze_btn.setFixedWidth(260); self.analyze_btn.setVisible(False)
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.progress = QtWidgets.QProgressBar(); self.progress.setFixedWidth(420)
        self.progress.setVisible(False)
        self.status_lbl = QtWidgets.QLabel(""); self.status_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.status_lbl.setWordWrap(True)
        # Constrain wrapping to the progress-bar width.  Without an explicit
        # width, QLabel's word-wrapped size hint can be too short on macOS and
        # clip the final line of longer segmentation status messages.
        self.status_lbl.setFixedWidth(420)
        self.status_lbl.setMinimumHeight(42)
        self.status_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        for widget in (title, sub, btn, self.past_runs_btn, self.seg_url_edit,
                       self.local_seg_cb, self.res_combo, self.setup_local_btn,
                       self.scout_cb, self.roi_mode_combo,
                       self.select_all_cb, self.study_list,
                       self.analyze_btn, self.progress, self.status_lbl):
            v.addWidget(widget, alignment=QtCore.Qt.AlignHCenter)
        v.addStretch()
        return w

    def _review_page(self):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)

        # left sidebar: levels + decision + export
        side = QtWidgets.QVBoxLayout()
        self.back_btn = QtWidgets.QPushButton("\u2190 Back to results")
        self.back_btn.clicked.connect(self._back_to_results)
        self.back_btn.setVisible(False)
        side.addWidget(self.back_btn)
        self.seg_banner = QtWidgets.QLabel("")
        self.seg_banner.setWordWrap(True)
        self.seg_banner.setVisible(False)
        side.addWidget(self.seg_banner)
        self.cal_label = QtWidgets.QLabel("")
        self.cal_label.setWordWrap(True)
        self.cal_label.setStyleSheet("color:#9a9aa2; font-size:11px;")
        side.addWidget(self.cal_label)
        side.addWidget(QtWidgets.QLabel("Vertebral levels"))
        self.level_list = QtWidgets.QListWidget(); self.level_list.setFixedWidth(240)
        self.level_list.currentRowChanged.connect(self._on_level_changed)
        side.addWidget(self.level_list)
        self.hu_label = QtWidgets.QLabel("-"); self.hu_label.setWordWrap(True)
        self.hu_label.setStyleSheet("background:#202024;padding:8px;border-radius:4px;")
        side.addWidget(self.hu_label)

        acc = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton("Include result"); self.accept_btn.clicked.connect(lambda: self._decide(True))
        self.reject_btn = QtWidgets.QPushButton("Exclude result"); self.reject_btn.clicked.connect(lambda: self._decide(False))
        acc.addWidget(self.accept_btn); acc.addWidget(self.reject_btn)
        side.addLayout(acc)
        self.finalize_btn = QtWidgets.QPushButton("Finalize (save review)")
        self.finalize_btn.clicked.connect(self.finalize)
        side.addWidget(self.finalize_btn)
        export_btn = QtWidgets.QPushButton("Export results..."); export_btn.clicked.connect(self.export)
        side.addWidget(export_btn)
        side.addStretch()
        h.addLayout(side)

        # center: tri-planar
        grid = QtWidgets.QGridLayout()
        self.ax = PlaneView("AXIAL"); self.sag = PlaneView("SAGITTAL"); self.cor = PlaneView("CORONAL")
        for p in (self.ax, self.sag, self.cor):
            p.moved.connect(self._make_click_handler(p))
        grid.addWidget(self.ax, 0, 0); grid.addWidget(self.sag, 0, 1); grid.addWidget(self.cor, 1, 0)
        ctrl = self._controls_box(); grid.addWidget(ctrl, 1, 1)
        cw = QtWidgets.QWidget(); cw.setLayout(grid); h.addWidget(cw, 1)
        return w

    # ---- results / past-runs grid ---------------------------------------
    def _results_page(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.results_back_btn = QtWidgets.QPushButton("\u2190 Back")
        self.results_back_btn.setFixedWidth(120)
        self.results_back_btn.clicked.connect(self._results_back)
        top.addWidget(self.results_back_btn)
        self.results_title = QtWidgets.QLabel("Results")
        self.results_title.setObjectName("title")
        top.addWidget(self.results_title); top.addStretch()
        v.addLayout(top)

        self.results_scroll = QtWidgets.QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_inner = QtWidgets.QWidget()
        self.results_vbox = QtWidgets.QVBoxLayout(self.results_inner)
        self.results_vbox.setAlignment(QtCore.Qt.AlignTop)
        self.results_scroll.setWidget(self.results_inner)
        v.addWidget(self.results_scroll)
        return w

    def _clear_results_grid(self):
        while self.results_vbox.count():
            item = self.results_vbox.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.deleteLater()

    def _show_grid(self, entries, title, back):
        """Render a list of cards. Each entry is a dict:
        {"kind": "batch", "batch": <batch dict>} or
        {"kind": "run", "run": <run dict>}."""
        self.results_title.setText(title)
        self._results_back_target = back
        self._clear_results_grid()
        if not entries:
            self.results_vbox.addWidget(QtWidgets.QLabel("No runs yet."))
        for entry in entries:
            self.results_vbox.addWidget(self._make_card(entry))
        self.stack.setCurrentIndex(2)

    def _make_card(self, entry):
        btn = QtWidgets.QPushButton()
        btn.setStyleSheet("text-align:left; padding:12px;")
        if entry["kind"] == "batch":
            b = entry["batch"]
            n = len(b.get("run_ids", []))
            btn.setText(f"Batch \u00b7 {n} studies \u00b7 {b.get('created', '')}")
            btn.clicked.connect(lambda _=False, bid=b["id"]: self.show_batch(bid))
        else:
            r = entry["run"]
            st = r.get("study", {})
            status = r.get("status", "ready")
            label = (st.get("study_description") or st.get("patient_label")
                     or st.get("series_uid", "study"))
            date = st.get("study_date", "")
            mark = {"ready": "", "reviewed": "  [reviewed]",
                    "failed": "  [FAILED]"}.get(status, "")
            btn.setText(f"{label}   {date}{mark}")
            if status == "failed":
                btn.setToolTip(r.get("error", "") or "analysis failed")
                btn.setStyleSheet("text-align:left; padding:12px; color:#c5544a;")
            btn.clicked.connect(lambda _=False, rid=r["id"]: self.open_run(rid))
        return btn

    def open_past_runs(self):
        """Top-level history: batches and standalone single runs, newest first."""
        self._grid_context = ("past", None)
        self._reshow_current_grid()

    def show_batch(self, batch_id):
        self._grid_context = ("batch", batch_id)
        self._reshow_current_grid()

    def _reshow_current_grid(self):
        """(Re)build the current grid from fresh store data so status changes
        (e.g. a newly [reviewed] run) are reflected."""
        ctx = getattr(self, "_grid_context", None)
        if ctx is None:
            self.stack.setCurrentIndex(0)
            return
        kind, arg = ctx
        if kind == "batch":
            try:
                batch = run_store.load_batch(arg)
            except (OSError, ValueError):
                self.stack.setCurrentIndex(0)
                return
            runs = []
            for rid in batch.get("run_ids", []):
                try:
                    runs.append(run_store.load_run(rid))
                except (OSError, ValueError):
                    continue
            entries = [{"kind": "run", "run": r} for r in runs]
            self._show_grid(entries, f"Batch \u00b7 {len(runs)} studies", back="past")
        else:  # past runs
            batches = run_store.list_batches()
            runs = run_store.list_runs()
            batched_ids = {rid for b in batches for rid in b.get("run_ids", [])}
            entries = [{"kind": "batch", "batch": b, "created": b.get("created", "")}
                       for b in batches]
            entries += [{"kind": "run", "run": r, "created": r.get("created", "")}
                        for r in runs if r["id"] not in batched_ids]
            entries.sort(key=lambda e: e.get("created", ""), reverse=True)
            self._show_grid(entries, "Past runs", back="landing")

    def open_run(self, run_id):
        try:
            state = run_store.reopen_run(run_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cannot open run", str(e))
            return
        self.load_state(state, run_id=run_id)

    def _results_back(self):
        target = getattr(self, "_results_back_target", "landing")
        if target == "past":
            self.open_past_runs()
        else:
            self.stack.setCurrentIndex(0)

    def _back_to_results(self):
        if getattr(self, "_grid_context", None) is None:
            self.stack.setCurrentIndex(0)
            return
        self._reshow_current_grid()

    def _controls_box(self):
        box = QtWidgets.QGroupBox("Display & ROI"); v = QtWidgets.QVBoxLayout(box)
        # window/level presets
        wl = QtWidgets.QHBoxLayout(); wl.addWidget(QtWidgets.QLabel("Window:"))
        for name, (c, w_) in {"Bone": (400, 1500), "Soft tissue": (40, 400)}.items():
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _=False, c=c, w_=w_: self._set_wl(c, w_))
            wl.addWidget(b)
        v.addLayout(wl)
        # overlay toggles
        ov = QtWidgets.QHBoxLayout()
        self.cb_body = QtWidgets.QCheckBox("Body"); self.cb_body.setChecked(True)
        self.cb_inner = QtWidgets.QCheckBox("Inner"); self.cb_inner.setChecked(True)
        self.cb_roi = QtWidgets.QCheckBox("ROI"); self.cb_roi.setChecked(True)
        for cb in (self.cb_body, self.cb_inner, self.cb_roi):
            cb.stateChanged.connect(self._toggle_overlays); ov.addWidget(cb)
        v.addLayout(ov)
        # radius slider
        v.addWidget(QtWidgets.QLabel("ROI radius (mm)"))
        rl = QtWidgets.QHBoxLayout()
        self.radius_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.radius_slider.setRange(20, 160)   # 2.0 - 16.0 mm (x10)
        self.radius_slider.valueChanged.connect(self._on_radius)
        self.radius_slider.sliderPressed.connect(self._radius_press)
        self.radius_slider.sliderReleased.connect(self._radius_release)
        self.radius_val = QtWidgets.QLabel("-")
        rl.addWidget(self.radius_slider); rl.addWidget(self.radius_val)
        v.addLayout(rl)
        self.recompute_btn = QtWidgets.QPushButton("Recompute (reset to automatic)")
        self.recompute_btn.clicked.connect(self._recompute)
        v.addWidget(self.recompute_btn)
        v.addStretch()
        return box

    # ---- actions ---------------------------------------------------------
    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open DICOM folder")
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder):
        """Discover studies in a folder and populate the multi-select chooser.

        Handles the realistic physician upload: a folder that may hold one study
        or many patients/studies in arbitrarily nested layouts. One checkable row
        per study is shown so any subset (or all) can be batch-processed; the
        best study is checked by default. Returns the best series (testability /
        backward compatibility).
        """
        self.folder = folder
        from ..io.study_discovery import discover_studies
        studies = discover_studies(folder, axial_only=True)
        self._studies = studies

        self.study_list.blockSignals(True)
        self.study_list.clear()
        n_patients = len({s.patient_id for s in studies})
        for i, study in enumerate(studies):
            prefix = (f"{study.patient_label}  \u2014  " if n_patients > 1 else "")
            it = QtWidgets.QListWidgetItem(prefix + study.option_label)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setData(QtCore.Qt.UserRole, i)
            it.setCheckState(QtCore.Qt.Checked if i == 0 else QtCore.Qt.Unchecked)
            self.study_list.addItem(it)
        self.study_list.blockSignals(False)

        has = bool(studies)
        self.study_list.setVisible(has)
        self.select_all_cb.setVisible(has)
        self.analyze_btn.setVisible(has)
        if has:
            self.status_lbl.setText(
                f"{len(studies)} study(ies) across {n_patients} patient(s). "
                "Check the studies to analyze (or Select all), then Analyze.")
        else:
            self.status_lbl.setText("No DICOM CT series found in this folder.")
        # backward-compat mapping used by tests / single-study callers
        self._row_series = {i: s.best_series for i, s in enumerate(studies)}
        self._refresh_scout_state()
        return studies[0].best_series if studies else None

    def _on_select_all(self, checked):
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        self.study_list.blockSignals(True)
        for i in range(self.study_list.count()):
            self.study_list.item(i).setCheckState(state)
        self.study_list.blockSignals(False)

    def _on_study_item_changed(self, _item):
        # keep the "Select all" box in sync without recursing
        total = self.study_list.count()
        checked = sum(1 for i in range(total)
                      if self.study_list.item(i).checkState() == QtCore.Qt.Checked)
        self.select_all_cb.blockSignals(True)
        self.select_all_cb.setChecked(total > 0 and checked == total)
        self.select_all_cb.blockSignals(False)
        # scout availability is per study, so re-check for the new selection
        self._refresh_scout_state()

    def _checked_studies(self):
        out = []
        for i in range(self.study_list.count()):
            it = self.study_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                idx = it.data(QtCore.Qt.UserRole)
                if 0 <= idx < len(self._studies):
                    out.append(self._studies[idx])
        return out

    def _set_scout_available(self, available: bool):
        """Enable/disable the scout checkbox and say why when it is off."""
        self.scout_cb.setEnabled(available)
        if available:
            self.scout_cb.setText("Record scout film measurements")
        else:
            self.scout_cb.setChecked(False)
            self.scout_cb.setText(
                "Record scout film measurements  (no scout films in this study)")

    def _refresh_scout_state(self):
        """Check the currently checked studies for localizers (header-only)."""
        from .analysis import has_scouts
        studies = self._checked_studies() or getattr(self, "_studies", [])
        available = bool(self.folder) and any(
            has_scouts(self.folder, series=st.best_series,
                       candidates=st.all_series)
            for st in studies)
        was_enabled = self.scout_cb.isEnabled()
        self._set_scout_available(available)
        if available and not was_enabled:
            self.scout_cb.setChecked(True)

    def _on_local_toggled(self, checked):
        # The server URL is irrelevant when segmenting locally; grey it out so the
        # choice is unambiguous.
        self.seg_url_edit.setEnabled(not checked)

    def _refresh_local_seg_state(self, prefer_local: bool = False):
        """Reflect whether local segmentation is installed: enable the checkbox
        if so, otherwise offer the one-time setup button."""
        avail = local_seg_available()
        self.local_seg_cb.setEnabled(avail)
        if avail and prefer_local:
            self.local_seg_cb.setChecked(True)
        if not avail and self.local_seg_cb.isChecked():
            self.local_seg_cb.setChecked(False)
        self.setup_local_btn.setVisible(not avail)

    def setup_local_seg(self):
        """Kick off the one-time local-segmentation install in a worker thread."""
        if QtWidgets.QMessageBox.question(
                self, "Set up local segmentation",
                "This downloads everything needed to segment on this computer -- "
                "a private Python, PyTorch + TotalSegmentator, and the model "
                "(about 1-3 GB total). Nothing needs to be preinstalled. It runs "
                "once and needs an internet connection. Continue?") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.setup_local_btn.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 0)
        self.status_lbl.setText("Setting up local segmentation...")
        self.setup_worker = SetupWorker()
        self.setup_worker.progress.connect(
            lambda m: self.status_lbl.setText(m))
        self.setup_worker.done.connect(self._on_setup_done)
        self.setup_worker.failed.connect(self._on_setup_failed)
        self.setup_worker.start()

    def _on_setup_done(self):
        self.progress.setVisible(False); self.progress.setRange(0, 100)
        self.setup_local_btn.setEnabled(True)
        self.status_lbl.setText("Local segmentation installed.")
        self._refresh_local_seg_state()
        self.local_seg_cb.setChecked(True)

    def _on_setup_failed(self, err):
        self.progress.setVisible(False); self.progress.setRange(0, 100)
        self.setup_local_btn.setEnabled(True)
        self.status_lbl.setText("Local setup failed.")
        self._show_error_dialog(err)

    def start_analyze(self):
        studies = self._checked_studies()
        if not studies:
            self.status_lbl.setText("Check at least one study to analyze.")
            return
        local = self.local_seg_cb.isChecked()
        fast = bool(self.res_combo.currentData())
        # Guard a full-res LOCAL run on a machine that likely lacks the RAM: the
        # segmentation runs on this computer only when local is checked, so the
        # crash risk (and the warning) only applies then.
        if local and not fast and not self._confirm_fullres_ram():
            return
        fast = bool(self.res_combo.currentData())   # may have changed in the dialog
        seg_url = self.seg_url_edit.text().strip() or None
        mode = self.roi_mode_combo.currentData() or DEFAULT_MODE
        # Remember the request so failure dialogs can offer one-click
        # "Use cloud instead" / "Try again" reruns.
        self._retry_ctx = {"studies": studies, "mode": mode, "seg_url": seg_url,
                           "local": local, "fast": fast}
        self.progress.setVisible(True); self.analyze_btn.setEnabled(False)

        if len(studies) == 1:
            self._start_single(studies[0], mode, seg_url, local, fast)
        else:
            self._start_batch(studies, mode, seg_url, local, fast)

    def _start_single(self, study, mode, seg_url, local, fast):
        self.series = study.best_series
        self._pending_study = study
        self._pending_backend = "local" if local else "cloud"
        self._pending_resolution = "fast" if fast else "full"
        self._pending_mode = mode
        self.worker = AnalyzeWorker(self.folder, self.series, mode,
                                    "physician", seg_url=seg_url, local=local,
                                    fast=fast, scout=self.scout_cb.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_state.connect(self._on_analyzed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _start_batch(self, studies, mode, seg_url, local, fast):
        self.progress.setRange(0, 0)
        self.batch_worker = BatchWorker(self.folder, studies, mode,
                                        seg_url=seg_url, local=local, fast=fast,
                                        scout=self.scout_cb.isChecked())
        self.batch_worker.progress.connect(self._on_progress)
        self.batch_worker.finished_batch.connect(self._on_batch_done)
        self.batch_worker.failed.connect(self._on_failed)
        self.batch_worker.start()

    def _on_batch_done(self, batch_id):
        self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False); self.progress.setRange(0, 100)
        self.show_batch(batch_id)

    def _confirm_fullres_ram(self) -> bool:
        """Warn before a full-res LOCAL run on a low-RAM machine.

        Returns True if analysis should proceed (possibly after switching to
        fast mode via the dialog), False if the user cancelled.
        """
        from ..segmentation.system_probe import fullres_ram_warning
        warning = fullres_ram_warning()
        if warning is None:
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Low memory for full-resolution")
        box.setText("Full-resolution local segmentation may crash on this computer")
        box.setInformativeText(warning)
        use_fast = box.addButton("Use fast (3 mm) instead",
                                 QtWidgets.QMessageBox.AcceptRole)
        anyway = box.addButton("Run full-res anyway",
                               QtWidgets.QMessageBox.DestructiveRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(use_fast)
        box.exec()
        clicked = box.clickedButton()
        if clicked is use_fast:
            # switch the selector to fast (index 1) and proceed
            self.res_combo.setCurrentIndex(1)
            return True
        if clicked is anyway:
            return True
        return False

    def _on_progress(self, msg, frac):
        if frac is None or frac < 0:          # indeterminate / busy
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(frac * 100))
        self.status_lbl.setText(msg)

    def _on_failed(self, err):
        self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.status_lbl.setText("Analysis failed.")
        self._show_error_dialog(err)

    # ---- structured error dialogs ----------------------------------------
    #
    # Every failure dialog has three layers: WHAT HAPPENED in plain language,
    # WHAT TO DO NOW (with live buttons when the app can do it itself), and a
    # "Show Details" expander with the technical detail + log path -- never a
    # raw traceback or command line as the headline.

    # kinds where "Repair local segmentation" / "Use cloud instead" make sense
    _LOCAL_KINDS = ("local-runtime-broken", "seg-crashed", "seg-oom", "setup-failed")
    _CLOUD_RETRY_KINDS = ("cloud-unreachable", "cloud-timeout", "cloud-failed")

    def _show_error_dialog(self, err):
        import traceback
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        if isinstance(err, UserFacingError):
            kind = err.kind
            box.setWindowTitle(err.title)
            box.setText(err.message)
            if err.remedy:
                box.setInformativeText(err.remedy)
            detail = err.detail
        else:
            kind = "unknown"
            box.setWindowTitle("Something went wrong")
            box.setText("Something went wrong that the app didn't anticipate.")
            box.setInformativeText(
                "Please try again. If it keeps happening, send us the log "
                f"file at:\n{applog.log_path()}")
            if isinstance(err, BaseException):
                detail = "".join(traceback.format_exception(err))
            else:
                detail = str(err)
        box.setDetailedText(f"{detail}\n\nLog file: {applog.log_path()}")

        can_rerun = bool(getattr(self, "_retry_ctx", None))
        repair_btn = cloud_btn = retry_btn = None
        if kind in self._LOCAL_KINDS:
            repair_btn = box.addButton("Repair local segmentation",
                                       QtWidgets.QMessageBox.ActionRole)
            if can_rerun:
                cloud_btn = box.addButton("Use cloud instead",
                                          QtWidgets.QMessageBox.AcceptRole)
        elif kind in self._CLOUD_RETRY_KINDS and can_rerun:
            retry_btn = box.addButton("Try again",
                                      QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec()
        clicked = box.clickedButton()
        if clicked is not None and clicked is repair_btn:
            self._start_repair()
        elif clicked is not None and clicked is cloud_btn:
            self._rerun_analysis(local=False)
        elif clicked is not None and clicked is retry_btn:
            self._rerun_analysis()

    def _start_repair(self):
        """Rebuild the managed local-seg env (the zero-argument 'fix it' path)."""
        self.stack.setCurrentIndex(0)
        self.setup_local_btn.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 0)
        self.status_lbl.setText("Repairing local segmentation...")
        self.setup_worker = SetupWorker(repair=True)
        self.setup_worker.progress.connect(lambda m: self.status_lbl.setText(m))
        self.setup_worker.done.connect(self._on_setup_done)
        self.setup_worker.failed.connect(self._on_setup_failed)
        self.setup_worker.start()

    def _rerun_analysis(self, local=None):
        """Re-run the last requested analysis, optionally forcing the cloud."""
        ctx = getattr(self, "_retry_ctx", None)
        if not ctx:
            return
        use_local = ctx["local"] if local is None else local
        if not use_local:
            self.local_seg_cb.setChecked(False)   # keep the UI honest
        self.stack.setCurrentIndex(0)
        self.progress.setVisible(True); self.analyze_btn.setEnabled(False)
        studies, mode = ctx["studies"], ctx["mode"]
        if len(studies) == 1:
            self._start_single(studies[0], mode, ctx["seg_url"], use_local,
                               ctx["fast"])
        else:
            self._start_batch(studies, mode, ctx["seg_url"], use_local,
                              ctx["fast"])

    def _on_analyzed(self, state):
        self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False)
        # Auto-save the single run into the library so it can be reopened later.
        run_id = None
        study = getattr(self, "_pending_study", None)
        try:
            meta = (study.as_meta() if study is not None
                    else run_store.study_meta_from_series(self.series, self.folder))
            rec = run_store.build_record(
                state, meta,
                backend=getattr(self, "_pending_backend", "cloud"),
                resolution=getattr(self, "_pending_resolution", "full"),
                mode=getattr(self, "_pending_mode", DEFAULT_MODE), status="ready")
            run_store.save_run(rec)
            run_id = rec["id"]
        except Exception:
            run_id = None            # persistence is best-effort; never block review
        self._grid_context = None     # single run: "Back to results" not applicable
        self.load_state(state, run_id=run_id)

    def load_state(self, state: ReviewState, run_id=None):
        """Populate the review screen from a ReviewState (also used in tests)."""
        self.state = state
        self.current_run_id = run_id
        self.back_btn.setVisible(getattr(self, "_grid_context", None) is not None)
        seg_status = state.case.get("seg_status")
        if seg_status and seg_status != "ok":
            reasons = (state.case.get("seg_check") or {}).get("global_reasons", [])
            self.seg_banner.setText(
                f"Segmentation {seg_status.upper()}: " + "; ".join(reasons) +
                ("  -- affected levels excluded." if seg_status == "invalid" else ""))
            color = "#c5544a" if seg_status == "invalid" else "#d8a23a"
            self.seg_banner.setStyleSheet(
                f"background:{color}; color:#111; padding:6px; border-radius:4px;")
            self.seg_banner.setVisible(True)
        else:
            self.seg_banner.setVisible(False)
        cal = state.case.get("calibration") or {}
        if cal:
            self.cal_label.setText(
                f"HU calibration factor {cal.get('factor')} "
                f"(kVp {cal.get('kvp')}); " + "; ".join(cal.get("notes", [])))
        self.level_list.clear()
        for lvl in state.results:
            it = QtWidgets.QListWidgetItem(lvl)
            it.setData(QtCore.Qt.UserRole, lvl)
            self.level_list.addItem(it)
        self._refresh_level_list_colors()
        self.stack.setCurrentIndex(1)
        if self.level_list.count():
            self.level_list.setCurrentRow(0)

    def _refresh_level_list_colors(self):
        for i in range(self.level_list.count()):
            it = self.level_list.item(i)
            r = self.state.results[it.data(QtCore.Qt.UserRole)]
            st = r.qc.get("qc_status")
            color = STATUS_COLOR.get(st, "#888888") if r.included else STATUS_COLOR["excluded"]
            mark = {"pass": "PASS", "review": "REVIEW", "fail": "FAIL",
                    "excluded": "EXCLUDED"}.get(st, "")
            dec = (" [included]" if r.accepted is True else
                   " [excluded]" if not r.included else "")
            if not r.editable:
                it.setText(f"{r.level:5s}  --  {mark}")
            else:
                med = r.stats.get("median_HU", float("nan"))
                it.setText(f"{r.level:5s}  {med:.0f} HU  {mark}{dec}")
            it.setForeground(QtGui.QColor(color))

    def _is_editable(self, level) -> bool:
        r = self.state.results.get(level) if self.state else None
        return bool(r is not None and r.editable)

    def _on_level_changed(self, row):
        if row < 0 or not self.state:
            return
        self.current_level = list(self.state.results.keys())[row]
        r = self.state.results[self.current_level]
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(int(round(r.radius_mm * 10)))
        self.radius_slider.blockSignals(False)
        editable = self._is_editable(self.current_level)
        self.accept_btn.setEnabled(editable)
        self.reject_btn.setEnabled(editable)
        self.radius_slider.setEnabled(editable)
        self.recompute_btn.setEnabled(editable)
        self.render()

    def _set_wl(self, c, w):
        self.wc, self.ww = c, w; self.render()

    def _toggle_overlays(self):
        self.show_body = self.cb_body.isChecked()
        self.show_inner = self.cb_inner.isChecked()
        self.show_roi = self.cb_roi.isChecked()
        self.render()

    def _radius_press(self):
        if self.current_level and self.state and \
                self.state.results[self.current_level].body_mask is not None:
            self.state.push_undo(self.current_level)
        self._radius_active = True

    def _radius_release(self):
        self._radius_active = False

    def _on_radius(self, val):
        if not self.current_level:
            return
        if self.state.results[self.current_level].body_mask is None:
            return                       # excluded level: nothing to resize
        if not self._radius_active:      # keyboard/programmatic step: its own undo
            self.state.push_undo(self.current_level)
        self.state.set_radius(self.current_level, val / 10.0)
        self._after_edit()

    def _recompute(self):
        if not self.current_level or not self._is_editable(self.current_level):
            return None
        self.state.push_undo(self.current_level)
        result = self.state.recompute_level(self.current_level)
        if result is None:
            self._undo()
            return None
        r = self.state.results[self.current_level]
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(int(round(r.radius_mm * 10)))
        self.radius_slider.blockSignals(False)
        self._after_edit()
        return r

    def _decide(self, accepted):
        if self.current_level and self._is_editable(self.current_level):
            self.state.push_undo(self.current_level)
            self.state.set_decision(self.current_level, accepted)
            self._refresh_level_list_colors()

    def _accept_advance(self):
        """Enter: include the current level (if measurable), then advance.
        Hard-excluded levels have no editable measurement and are skipped."""
        if not self.state or self.stack.currentIndex() != 1 or not self.current_level:
            return
        if self._is_editable(self.current_level):
            self._decide(True)
        row = self.level_list.currentRow()
        if 0 <= row < self.level_list.count() - 1:
            self.level_list.setCurrentRow(row + 1)

    def _undo(self):
        if not self.state or not self.state.can_undo():
            return
        level = self.state.undo()
        if level is None:
            return
        self._select_level(level)        # focus the reverted level
        r = self.state.results.get(level)
        if r is not None and r.body_mask is not None:
            self.radius_slider.blockSignals(True)   # re-sync slider to restored radius
            self.radius_slider.setValue(int(round(r.radius_mm * 10)))
            self.radius_slider.blockSignals(False)
        self._after_edit()

    def _select_level(self, level):
        for i in range(self.level_list.count()):
            if self.level_list.item(i).data(QtCore.Qt.UserRole) == level:
                self.level_list.setCurrentRow(i)   # triggers _on_level_changed
                return
        # not in the list (shouldn't happen): at least sync current + slider
        self.current_level = level
        r = self.state.results.get(level)
        if r is not None:
            self.radius_slider.blockSignals(True)
            self.radius_slider.setValue(int(round(r.radius_mm * 10)))
            self.radius_slider.blockSignals(False)

    def _after_edit(self):
        self._refresh_level_list_colors()
        self.render()

    def _make_click_handler(self, plane):
        def handler(col_mm, row_mm, record=True):
            if not self.current_level:
                return
            r = self.state.results[self.current_level]
            if r.body_mask is None:
                return                   # excluded level: no ROI to move
            if not self._drag_active:    # snapshot once, before the gesture's first change
                self.state.push_undo(self.current_level)
            sh, sv = plane.sh, plane.sv
            col = int(round(col_mm / sh)); row = int(round(row_mm / sv))
            cx, cy, cz = r.center_idx
            if plane is self.ax:      # cols=X, rows=Y
                cx, cy = col, row
            elif plane is self.sag:   # cols=Y, rows=Z
                cy, cz = col, row
            elif plane is self.cor:   # cols=X, rows=Z
                cx, cz = col, row
            self.state.set_center(self.current_level, (cx, cy, cz), record=record)
            self._drag_active = not record    # True mid-drag, False on release/click
            self._after_edit()
        return handler

    # ---- rendering -------------------------------------------------------
    def _display_arrays(self, r):
        """(hu_crop, body, inner, roi, center_idx) for the tri-planar views.

        Excluded levels still show the CT + segmentation outline so the
        physician can review the anatomy, but NO measurement ROI is drawn (there
        is no valid measurement). Gate-excluded levels that have no measurement
        crop get one built straight from the segmentation label.
        """
        excluded = not r.editable
        if r.crop_slices is not None and r.body_mask is not None:
            hu = self.state.volume.hu[r.crop_slices]
            roi = None if excluded else r.roi_mask
            inner = None if excluded else r.inner_mask
            return hu, r.body_mask, inner, roi, r.center_idx
        # gate-excluded: derive a crop straight from the segmentation label
        from ..geometry.coords import crop_bbox
        from ..segmentation.totalseg_runner import label_id_for
        try:
            lab = label_id_for(r.level)
        except KeyError:
            return None
        mask = self.state.seg == lab
        if not mask.any():
            return None
        sl = crop_bbox(mask, self.state.volume.spacing)
        body = mask[sl]
        center = tuple(int(round(c)) for c in np.argwhere(body).mean(axis=0))
        return self.state.volume.hu[sl], body, None, None, center

    def render(self):
        if not self.state or not self.current_level:
            return
        r = self.state.results[self.current_level]
        disp = self._display_arrays(r)
        if disp is None:
            for p in (self.ax, self.sag, self.cor):
                p.img.clear()
            self._update_stats_panel(r)
            return
        hu, body, inner, roi, (cx, cy, cz) = disp
        sx, sy, sz = self.state.volume.spacing
        for p in (self.ax, self.sag, self.cor):
            p._wl = (self.wc, self.ww)

        def layers(b, i, o):
            out = []
            if self.show_body and b is not None: out.append((b, COL_BODY, 0.25))
            if self.show_inner and i is not None: out.append((i, COL_INNER, 0.30))
            if self.show_roi and o is not None: out.append((o, COL_ROI, 0.55))
            return out

        def sl(arr, idx, axis):
            if arr is None:
                return None
            return (arr[:, :, idx] if axis == 2 else
                    arr[idx, :, :] if axis == 0 else arr[:, idx, :]).T

        # re-fit the view only when a new level is shown; keep zoom during edits
        reset = (self.current_level != getattr(self, "_last_rendered_level", None))
        # AXIAL z=cz: rows=Y, cols=X
        self.ax.set_data(hu[:, :, cz].T,
                         layers(sl(body, cz, 2), sl(inner, cz, 2), sl(roi, cz, 2)),
                         sx, sy, (cx, cy), labels=("R", "L", "A", "P"), reset_range=reset)
        # SAGITTAL x=cx: rows=Z, cols=Y
        self.sag.set_data(hu[cx, :, :].T,
                          layers(sl(body, cx, 0), sl(inner, cx, 0), sl(roi, cx, 0)),
                          sy, sz, (cy, cz), labels=("A", "P", "S", "I"), reset_range=reset)
        # CORONAL y=cy: rows=Z, cols=X
        self.cor.set_data(hu[:, cy, :].T,
                          layers(sl(body, cy, 1), sl(inner, cy, 1), sl(roi, cy, 1)),
                          sx, sz, (cx, cz), labels=("R", "L", "S", "I"), reset_range=reset)
        self._last_rendered_level = self.current_level
        self._update_stats_panel(r)

    @staticmethod
    def _habitus_line(stats) -> str:
        """One-line body habitus summary, or '' when no scout values exist."""
        w, d = stats.get("body_width_lr_mm"), stats.get("body_depth_ap_mm")
        if w is None and d is None:
            return ""
        parts = []
        if w is not None:
            parts.append(f"width {w:.0f} mm (LR)")
        if d is not None:
            parts.append(f"depth {d:.0f} mm (AP)")
        eff = stats.get("body_effective_diameter_mm")
        if eff is not None:
            parts.append(f"eff. diam {eff:.0f} mm")
        return ("<span style='color:#8a8a92'>body habitus (scout): "
                + " | ".join(parts) + "</span><br>")

    def _update_stats_panel(self, r):
        s = r.stats
        warns = "<br>".join(f"&bull; {w}" for w in r.qc.get("warnings", [])) or "none"
        habitus = self._habitus_line(s)
        self.radius_val.setText(f"{r.radius_mm:.1f}")
        if not r.editable or not s:
            self.hu_label.setText(
                f"<b>{r.level}</b> ({MODE_LABELS.get(r.mode, r.mode)})<br>"
                f"<b style='color:#c5544a'>EXCLUDED</b> - no HU reported<br>"
                f"reason: {r.qc.get('exclusion_reason', '')}<br>{habitus}{warns}")
            return
        cal = s.get("calibrated_median_HU")
        cal_line = (f"calibrated median {cal:.0f} HU<br>"
                    if cal is not None else "")
        ctx = s.get("hu_context")
        ctx_line = f"<span style='color:#8a8a92'>{ctx}</span><br>" if ctx else ""
        inclusion = ("INCLUDED" if r.included else "EXCLUDED FROM RESULTS")
        reason = (f"reason: {r.exclusion_reason()}<br>"
                  if not r.included else "")
        self.hu_label.setText(
            f"<b>{r.level}</b> ({MODE_LABELS.get(r.mode, r.mode)})<br>"
            f"median {s.get('median_HU', float('nan')):.0f} | mean {s.get('mean_HU', float('nan')):.0f} HU<br>"
            f"{cal_line}"
            f"SD {s.get('sd_HU', float('nan')):.0f} | p05-p95 {s.get('p05_HU', float('nan')):.0f}-{s.get('p95_HU', float('nan')):.0f}<br>"
            f"radius {r.radius_mm:.1f} mm | clearance {r.margin_mm:.1f} mm<br>"
            f"volume {s.get('volume_mm3', 0):.0f} mm&sup3;<br>"
            f"{habitus}"
            f"<b>QC: {r.qc.get('qc_status')}</b> | <b>{inclusion}</b><br>"
            f"{reason}{ctx_line}{warns}")

    def finalize(self):
        """Mark the run reviewed: save the physician's edits + write the export
        into the run library so it persists and shows as reviewed in past runs."""
        if not self.state:
            return
        if not self.current_run_id:
            QtWidgets.QMessageBox.information(
                self, "Nothing to finalize",
                "This view is not linked to a saved run.")
            return
        try:
            rec = run_store.load_run(self.current_run_id)
            rec["overrides"] = run_store.overrides_from_state(self.state)
            rec["measurements"] = {lvl: r.summary()
                                   for lvl, r in self.state.results.items()}
            rec["schema_version"] = run_store.RUN_SCHEMA_VERSION
            rec["status"] = "reviewed"
            export_dir = os.path.join(run_store.run_dir(self.current_run_id), "export")
            self._do_export(export_dir)
            rec["export_dir"] = export_dir
            run_store.save_run(rec)
        except Exception as e:
            applog.log_exception("finalize failed", e)
            self._show_error_dialog(e)
            return
        QtWidgets.QMessageBox.information(
            self, "Run finalized",
            "Review saved. This study is marked reviewed and its results were "
            "written to the run library.")

    def export(self):
        if not self.state:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Export to folder")
        if not out:
            return
        self._do_export(out)
        QtWidgets.QMessageBox.information(self, "Export complete",
                                          f"Results written to:\n{out}")

    def _do_export(self, out):
        """Run the export (no dialog/message box -- testable)."""
        from ..export.writers import export_case
        return export_case(self.state.case, self.state.volume, out)


def _app_icon():
    """Locate the bundled app icon (works frozen and from source)."""
    import sys
    candidates = []
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidates.append(os.path.join(base, "icons", "spine_hu.png"))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates.append(os.path.join(repo_root, "packaging", "icons", "spine_hu.png"))
    for path in candidates:
        if os.path.exists(path):
            return QtGui.QIcon(path)
    return None


def main():
    import sys
    applog.init_logging()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Spine HU Tool")
    icon = _app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
