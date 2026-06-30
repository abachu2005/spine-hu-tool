"""Headless end-to-end GUI smoke test on the REAL test folder: runs the exact
analyze->load_state->render path the desktop app uses, exercises every level
(measured + excluded), and saves a screenshot."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.pop("SPINE_HU_SEG_URL", None)        # use the cached full-res seg
import time
import traceback

from PySide6 import QtWidgets, QtCore
from spine_hu_tool.app.analysis import analyze_dataset
from spine_hu_tool.app.viewer import MainWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

t0 = time.time()
state = analyze_dataset("Test Data/100007EC", mode="centroid_volume_sphere",
                        progress=lambda m, f: print(f"[{f*100:4.0f}%] {m}", flush=True))
print(f"analyze_dataset finished in {time.time()-t0:.0f}s", flush=True)

win = MainWindow(); win.resize(1500, 950)
win.load_state(state)
win.show(); app.processEvents()

print(f"page index (1=review): {win.stack.currentIndex()}")
print(f"seg banner visible={win.seg_banner.isVisible()} text={win.seg_banner.text()[:90]!r}")
print(f"calibration label={win.cal_label.text()[:90]!r}")
print(f"levels listed: {win.level_list.count()}")

errors = []
measured_row = None
for i in range(win.level_list.count()):
    win.level_list.setCurrentRow(i)
    lvl = win.level_list.item(i).data(QtCore.Qt.UserRole)
    r = state.results[lvl]
    try:
        win.render()
        app.processEvents()
    except Exception:
        errors.append((lvl, traceback.format_exc()))
    status = r.qc.get("qc_status")
    if measured_row is None and status != "excluded":
        measured_row = i
    print(f"  {lvl:4} status={status:9} listtext={win.level_list.item(i).text()!r}")

print(f"render errors: {len(errors)}")
for lvl, tb in errors:
    print(f"--- {lvl} ---\n{tb}")

if measured_row is not None:
    win.level_list.setCurrentRow(measured_row); win.render(); app.processEvents()
out = "work/gui_smoke_real.png"
win.grab().save(out)
print(f"saved screenshot -> {out}")
print("GUI SMOKE OK" if not errors else "GUI SMOKE HAD ERRORS")
