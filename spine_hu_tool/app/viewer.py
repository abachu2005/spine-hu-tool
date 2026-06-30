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
    failed = QtCore.Signal(str)

    def __init__(self, folder, series, mode, reviewer, seg_url=None, api_key=None,
                 local=False):
        super().__init__()
        self.folder, self.series, self.mode, self.reviewer = folder, series, mode, reviewer
        self.seg_url, self.api_key, self.local = seg_url, api_key, local

    def run(self):
        try:
            st = analyze_dataset(self.folder, series=self.series, mode=self.mode,
                                 only_clean=True, reviewer=self.reviewer,
                                 seg_url=self.seg_url, api_key=self.api_key,
                                 local=self.local,
                                 progress=lambda m, f: self.progress.emit(m, f))
            self.finished_state.emit(st)
        except Exception as e:  # surfaced to the user
            self.failed.emit(str(e))


class SetupWorker(QtCore.QThread):
    """Installs the local-segmentation runtime (torch + TotalSegmentator +
    weights) off the UI thread, streaming progress lines."""
    progress = QtCore.Signal(str)
    done = QtCore.Signal()
    failed = QtCore.Signal(str)

    def run(self):
        try:
            from ..segmentation.local_setup import setup_local_seg
            setup_local_seg(progress=lambda m: self.progress.emit(m))
            self.done.emit()
        except Exception as e:  # surfaced to the user
            self.failed.emit(str(e))


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
        # display state
        self.wc, self.ww = 400.0, 1500.0
        self.show_body = True
        self.show_inner = True
        self.show_roi = True
        self._drag_active = False     # coalesce a cursor drag into one undo step
        self._radius_active = False   # coalesce a slider sweep into one undo step
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
            "Segmentation runs on a shared, unauthenticated cloud service: "
            "scans you open are uploaded to that hosted service for "
            "processing. Do NOT upload protected health information (PHI) — "
            "use de-identified data only.\n\n"
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
        # Segmentation runs on Cloud Run by default (prefilled); the heavy ML
        # step never touches this machine unless the field is explicitly cleared.
        self.seg_url_edit = QtWidgets.QLineEdit(resolve_seg_url(None) or "")
        self.seg_url_edit.setFixedWidth(420)
        self.seg_url_edit.setPlaceholderText("Segmentation server URL (cloud default)")
        # Local segmentation toggle: when checked, the heavy TotalSegmentator step
        # runs on this machine instead of the cloud service (requires the optional
        # local-seg dependencies / TotalSegmentator to be installed).
        self.local_seg_cb = QtWidgets.QCheckBox(
            "Run segmentation on this computer (no upload)")
        self.local_seg_cb.setToolTip(
            "Process the scan locally with TotalSegmentator instead of the cloud "
            "service. Needs more RAM/CPU; set it up once with the button below.")
        self.local_seg_cb.toggled.connect(self._on_local_toggled)
        # One-time installer for the local ML runtime (torch + TotalSegmentator
        # + weights) into a user-managed environment outside the app bundle.
        self.setup_local_btn = QtWidgets.QPushButton("Set up local segmentation...")
        self.setup_local_btn.setFixedWidth(420)
        self.setup_local_btn.clicked.connect(self.setup_local_seg)
        self._refresh_local_seg_state()
        self.roi_mode_combo = QtWidgets.QComboBox(); self.roi_mode_combo.setFixedWidth(420)
        for label, key in EXPOSED_MODES.items():
            self.roi_mode_combo.addItem(label, key)
        self.roi_mode_combo.setCurrentText(MODE_LABELS.get(DEFAULT_MODE, "centroid"))
        self.series_combo = QtWidgets.QComboBox(); self.series_combo.setFixedWidth(420)
        self.series_combo.setVisible(False)
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.setFixedWidth(260); self.analyze_btn.setVisible(False)
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.progress = QtWidgets.QProgressBar(); self.progress.setFixedWidth(420)
        self.progress.setVisible(False)
        self.status_lbl = QtWidgets.QLabel(""); self.status_lbl.setAlignment(QtCore.Qt.AlignCenter)
        for widget in (title, sub, btn, self.seg_url_edit, self.local_seg_cb,
                       self.setup_local_btn, self.roi_mode_combo, self.series_combo,
                       self.analyze_btn, self.progress, self.status_lbl):
            v.addWidget(widget, alignment=QtCore.Qt.AlignHCenter)
        v.addStretch()
        return w

    def _review_page(self):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)

        # left sidebar: levels + decision + export
        side = QtWidgets.QVBoxLayout()
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
        self.accept_btn = QtWidgets.QPushButton("Accept"); self.accept_btn.clicked.connect(lambda: self._decide(True))
        self.reject_btn = QtWidgets.QPushButton("Reject"); self.reject_btn.clicked.connect(lambda: self._decide(False))
        acc.addWidget(self.accept_btn); acc.addWidget(self.reject_btn)
        side.addLayout(acc)
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
        recompute = QtWidgets.QPushButton("Recompute (reset to automatic)")
        recompute.clicked.connect(self._recompute)
        v.addWidget(recompute)
        v.addStretch()
        return box

    # ---- actions ---------------------------------------------------------
    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open DICOM folder")
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder):
        """Series-select a folder and populate the chooser (no dialog -- testable).

        Handles folders that contain many patients/studies (the usual physician
        upload): series are grouped under disabled patient/study headers and the
        best axial CT is preselected.
        """
        self.folder = folder
        from ..io.series_selector import select_ct_series, list_studies
        best, cands = select_ct_series(folder)
        self.series_combo.clear()
        self._row_series = {}        # selectable combo row -> SeriesInfo
        patients = list_studies(cands, axial_only=True)
        model = self.series_combo.model()
        best_row = None
        multi_patient = len(patients) > 1
        for plabel, studies in patients:
            if multi_patient:                       # only show patient header when needed
                self.series_combo.addItem(f"\u2014  {plabel}")
                model.item(self.series_combo.count() - 1).setEnabled(False)
            for s in studies:
                self.series_combo.addItem(s.option_label)
                row = self.series_combo.count() - 1
                self._row_series[row] = s
                if best is not None and s.series_uid == best.series_uid:
                    best_row = row
        if best_row is not None:
            self.series_combo.setCurrentIndex(best_row)
        self.series_combo.setVisible(True); self.analyze_btn.setVisible(True)
        n_studies = len(self._row_series)
        if best is not None:
            self.status_lbl.setText(
                f"{n_studies} study(ies) found across {len(patients)} patient(s). "
                f"Selected: {best.study_description or best.description}")
        elif n_studies:
            self.status_lbl.setText("No clear axial CT auto-detected; pick a series manually.")
        else:
            self.status_lbl.setText("No DICOM CT series found in this folder.")
        return best

    def _on_local_toggled(self, checked):
        # The server URL is irrelevant when segmenting locally; grey it out so the
        # choice is unambiguous.
        self.seg_url_edit.setEnabled(not checked)

    def _refresh_local_seg_state(self):
        """Reflect whether local segmentation is installed: enable the checkbox
        if so, otherwise offer the one-time setup button."""
        avail = local_seg_available()
        self.local_seg_cb.setEnabled(avail)
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

    def _on_setup_failed(self, msg):
        self.progress.setVisible(False); self.progress.setRange(0, 100)
        self.setup_local_btn.setEnabled(True)
        self.status_lbl.setText("Local setup failed.")
        QtWidgets.QMessageBox.critical(self, "Local setup failed", msg)

    def start_analyze(self):
        series = getattr(self, "_row_series", {}).get(self.series_combo.currentIndex())
        if series is None:        # header/empty selection -> first available series
            series = next(iter(getattr(self, "_row_series", {}).values()), None)
        if series is None:
            self.status_lbl.setText("No selectable CT series in this folder.")
            return
        self.series = series
        self.progress.setVisible(True); self.analyze_btn.setEnabled(False)
        local = self.local_seg_cb.isChecked()
        seg_url = self.seg_url_edit.text().strip() or None
        mode = self.roi_mode_combo.currentData() or DEFAULT_MODE
        self.worker = AnalyzeWorker(self.folder, self.series, mode,
                                    "physician", seg_url=seg_url, local=local)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_state.connect(self._on_analyzed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, msg, frac):
        if frac is None or frac < 0:          # indeterminate / busy
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(frac * 100))
        self.status_lbl.setText(msg)

    def _on_failed(self, msg):
        self.analyze_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Analysis failed", msg)

    def _on_analyzed(self, state):
        self.analyze_btn.setEnabled(True)
        self.load_state(state)

    def load_state(self, state: ReviewState):
        """Populate the review screen from a ReviewState (also used in tests)."""
        self.state = state
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
            color = STATUS_COLOR.get(st, "#888888")
            mark = {"pass": "PASS", "review": "REVIEW", "fail": "FAIL",
                    "excluded": "EXCLUDED"}.get(st, "")
            dec = {True: " [accepted]", False: " [rejected]"}.get(r.accepted, "")
            if st == "excluded":
                it.setText(f"{r.level:5s}  --  {mark}")
            else:
                med = r.stats.get("median_HU", float("nan"))
                it.setText(f"{r.level:5s}  {med:.0f} HU  {mark}{dec}")
            it.setForeground(QtGui.QColor(color))

    def _is_excluded(self, level) -> bool:
        r = self.state.results.get(level) if self.state else None
        return bool(r is not None and r.qc.get("qc_status") == "excluded")

    def _on_level_changed(self, row):
        if row < 0 or not self.state:
            return
        self.current_level = list(self.state.results.keys())[row]
        r = self.state.results[self.current_level]
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(int(round(r.radius_mm * 10)))
        self.radius_slider.blockSignals(False)
        # excluded levels have no valid measurement: no accept/reject (or resize)
        excluded = self._is_excluded(self.current_level)
        self.accept_btn.setEnabled(not excluded)
        self.reject_btn.setEnabled(not excluded)
        self.radius_slider.setEnabled(not excluded)
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
        if not self.current_level:
            return None
        self.state.push_undo(self.current_level)
        self.state.recompute_level(self.current_level)
        r = self.state.results[self.current_level]
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(int(round(r.radius_mm * 10)))
        self.radius_slider.blockSignals(False)
        self._after_edit()
        return r

    def _decide(self, accepted):
        # excluded levels carry no measurement, so there's nothing to accept/reject
        if self.current_level and not self._is_excluded(self.current_level):
            self.state.push_undo(self.current_level)
            self.state.set_decision(self.current_level, accepted)
            self._refresh_level_list_colors()

    def _accept_advance(self):
        """Enter: accept the current level (if measurable), then move to the next.
        Excluded levels can't be accepted, so Enter just skips past them."""
        if not self.state or self.stack.currentIndex() != 1 or not self.current_level:
            return
        if not self._is_excluded(self.current_level):
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
        excluded = r.qc.get("qc_status") == "excluded"
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

    def _update_stats_panel(self, r):
        s = r.stats
        warns = "<br>".join(f"&bull; {w}" for w in r.qc.get("warnings", [])) or "none"
        self.radius_val.setText(f"{r.radius_mm:.1f}")
        if r.qc.get("qc_status") == "excluded":
            self.hu_label.setText(
                f"<b>{r.level}</b> ({MODE_LABELS.get(r.mode, r.mode)})<br>"
                f"<b style='color:#c5544a'>EXCLUDED</b> - no HU reported<br>"
                f"reason: {r.qc.get('exclusion_reason', '')}<br>{warns}")
            return
        cal = s.get("calibrated_median_HU")
        cal_line = (f"calibrated median {cal:.0f} HU<br>"
                    if cal is not None else "")
        ctx = s.get("hu_context")
        ctx_line = f"<span style='color:#8a8a92'>{ctx}</span><br>" if ctx else ""
        self.hu_label.setText(
            f"<b>{r.level}</b> ({MODE_LABELS.get(r.mode, r.mode)})<br>"
            f"median {s.get('median_HU', float('nan')):.0f} | mean {s.get('mean_HU', float('nan')):.0f} HU<br>"
            f"{cal_line}"
            f"SD {s.get('sd_HU', float('nan')):.0f} | p05-p95 {s.get('p05_HU', float('nan')):.0f}-{s.get('p95_HU', float('nan')):.0f}<br>"
            f"radius {r.radius_mm:.1f} mm | clearance {r.margin_mm:.1f} mm<br>"
            f"volume {s.get('volume_mm3', 0):.0f} mm&sup3;<br>"
            f"<b>QC: {r.qc.get('qc_status')}</b><br>{ctx_line}{warns}")

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
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Spine HU Tool")
    icon = _app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
