"""GUI user-test of the scout body-habitus feature on the real studies.

Drives the actual viewer (offscreen) the way a physician would: open a folder,
check the state of the "Record scout film measurements" checkbox, analyze, walk
the level list reading the stats panel, and export. Run:

    QT_QPA_PLATFORM=offscreen python work/gui_scout_user_test.py
"""
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from PySide6 import QtWidgets  # noqa: E402

STUDIES = ["Test Data/10000680", "Test Data/100007EC", "Test Data/Anon2"]


def strip_html(s):
    return re.sub(r"<[^>]+>", " ", s.replace("<br>", " | ")).strip()


def run_study(app, folder, out_root):
    from spine_hu_tool.app.viewer import MainWindow
    print("=" * 78)
    print(folder)
    win = MainWindow()
    win._load_folder(os.path.join(ROOT, folder))
    print(f"  checkbox: enabled={win.scout_cb.isEnabled()} "
          f"checked={win.scout_cb.isChecked()}  label={win.scout_cb.text()!r}")
    print(f"  status  : {win.status_lbl.text()}")

    win.start_analyze()
    while win.worker.isRunning():          # pump the event loop while it works
        app.processEvents()
        win.worker.wait(200)
    app.processEvents()
    if win.state is None:
        print("  ANALYSIS FAILED:", win.status_lbl.text())
        return

    block = win.state.case.get("scout") or {}
    print(f"  scout   : available={block.get('available')} "
          f"views={list(block.get('views') or {})} "
          f"warnings={block.get('warnings')}")

    for row in range(win.level_list.count()):
        win.level_list.setCurrentRow(row)
        app.processEvents()
        print("   ", strip_html(win.hu_label.text())[:170])

    out = os.path.join(out_root, os.path.basename(folder))
    written = win._do_export(out)
    print(f"  exported: {sorted(written)}")
    odir = written.get("overlays")
    if odir:
        pngs = sorted(f for f in os.listdir(odir) if f.startswith("scout_"))
        print(f"  scout overlays: {pngs}")


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    out_root = os.path.join(ROOT, "work", "scout_user_test")
    for folder in STUDIES:
        if not os.path.isdir(os.path.join(ROOT, folder)):
            print(f"skipping missing {folder}")
            continue
        run_study(app, folder, out_root)


if __name__ == "__main__":
    main()
