"""Task 1 (offline local run) safety tests.

CRITICAL: none of these run real local segmentation. They assert the constructed
environment / command / discovery logic, monkeypatching any subprocess so
TotalSegmentator is never actually launched.
"""
import os
import sys

import numpy as np

from spine_hu_tool.core import Volume
from spine_hu_tool.segmentation import local_setup as ls
from spine_hu_tool.segmentation import totalseg_runner as tr
from spine_hu_tool.segmentation import system_probe as sp


# --- env sanitation ---------------------------------------------------------

def test_child_env_strips_pyinstaller_loader_vars(monkeypatch):
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                "PYTHONHOME", "PYTHONPATH"):
        monkeypatch.setenv(var, "/frozen/leak")
    env = ls.child_env()
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                "PYTHONHOME", "PYTHONPATH"):
        assert var not in env, f"{var} leaked into child env"


def test_child_env_restores_pyinstaller_originals(monkeypatch):
    # PyInstaller stashes the pre-launch value in <VAR>_ORIG; child_env restores it.
    monkeypatch.setenv("LD_LIBRARY_PATH", "/frozen/bundle/libs")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system/original")
    env = ls.child_env()
    assert env["LD_LIBRARY_PATH"] == "/system/original"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_child_env_applies_extra_and_ignores_none():
    env = ls.child_env({"FOO": "bar", "SKIP": None})
    assert env["FOO"] == "bar"
    assert "SKIP" not in env


# --- bundled runtime discovery ----------------------------------------------

def test_bundled_env_discovery_via_meipass(monkeypatch, tmp_path):
    envdir = tmp_path / "localseg-env"
    bindir = envdir / ("Scripts" if sys.platform.startswith("win") else "bin")
    bindir.mkdir(parents=True)
    ts_name = "TotalSegmentator" + (".exe" if sys.platform.startswith("win") else "")
    (bindir / ts_name).write_text("#!/bin/sh\n")
    weights = tmp_path / "totalseg-weights"
    weights.mkdir()

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("SPINE_HU_BUNDLED_ENV", raising=False)
    monkeypatch.delenv("SPINE_HU_BUNDLED_WEIGHTS", raising=False)

    assert ls.bundled_env_dir() == str(envdir)
    assert ls.bundled_ts_binary() == str(bindir / ts_name)
    assert ls.bundled_weights_dir() == str(weights)
    assert ls.weights_home() == str(weights)


def test_bundled_env_overrides(monkeypatch, tmp_path):
    d = tmp_path / "custom-env"
    d.mkdir()
    monkeypatch.setenv("SPINE_HU_BUNDLED_ENV", str(d))
    assert ls.bundled_env_dir() == str(d)
    monkeypatch.setenv("SPINE_HU_BUNDLED_ENV", str(tmp_path / "does-not-exist"))
    assert ls.bundled_env_dir() is None


def test_ts_binary_prefers_bundled(monkeypatch, tmp_path):
    fake = str(tmp_path / "bin" / "TotalSegmentator")
    monkeypatch.setattr(ls, "bundled_ts_binary", lambda: fake)
    assert tr._ts_binary() == fake
    assert tr.local_seg_available() is True


def test_ts_binary_none_when_nothing_available(monkeypatch):
    import shutil
    monkeypatch.setattr(ls, "bundled_ts_binary", lambda: None)
    monkeypatch.setattr(ls, "managed_ts_binary", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    # no TotalSegmentator sitting next to this interpreter
    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path, "exists",
        lambda p: False if str(p).endswith("TotalSegmentator") else real_exists(p))
    assert tr._ts_binary() is None


# --- run_segmentation builds a sanitized env + weights, never launches -------

def test_run_segmentation_env_and_command(monkeypatch, tmp_path):
    """run_segmentation must build a sanitized env with TOTALSEG_HOME_DIR and a
    valid command -- captured via a fake subprocess so nothing is executed."""
    monkeypatch.setattr(tr, "_ts_binary", lambda: "/fake/TotalSegmentator")
    monkeypatch.setattr(ls, "weights_home", lambda: "/bundled/weights")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/frozen/leak")

    captured = {}

    class _Proc:
        returncode = 0

    def _fake_run(cmd, env=None, stdout=None, stderr=None):
        captured["cmd"] = cmd
        captured["env"] = env
        # emulate TS writing the expected output so no fallback path is taken
        out_idx = cmd.index("-o") + 1
        with open(cmd[out_idx], "wb") as fh:
            fh.write(b"")
        return _Proc()

    monkeypatch.setattr(tr.subprocess, "run", _fake_run)
    # load_segmentation would try to read the (empty) file; stub it out.
    monkeypatch.setattr(tr, "load_segmentation", lambda p: np.zeros((2, 2, 2), np.int16))
    # save_volume_nifti writes the input; stub to avoid real IO cost.
    monkeypatch.setattr(tr, "save_volume_nifti", lambda vol, path: open(path, "wb").close())

    vol = Volume(hu=np.zeros((4, 4, 4), dtype=np.int16), spacing=(1.0, 1.0, 1.0))
    tr.run_segmentation(vol, str(tmp_path), "case", fast=True)

    env = captured["env"]
    assert env["TOTALSEG_HOME_DIR"] == "/bundled/weights"
    assert env["TOTALSEG_WEIGHTS_PATH"] == "/bundled/weights"
    assert "DYLD_LIBRARY_PATH" not in env, "frozen loader var leaked into TS env"
    assert "--fast" in captured["cmd"]
    assert captured["cmd"][0] == "/fake/TotalSegmentator"


# --- RAM guard --------------------------------------------------------------

def test_fullres_ram_warning_low(monkeypatch):
    monkeypatch.setattr(sp, "total_ram_gb", lambda: 8.0)
    msg = sp.fullres_ram_warning()
    assert msg is not None and "8 GB" in msg


def test_fullres_ram_warning_high(monkeypatch):
    monkeypatch.setattr(sp, "total_ram_gb", lambda: 64.0)
    assert sp.fullres_ram_warning() is None


def test_fullres_ram_warning_unknown_is_silent(monkeypatch):
    monkeypatch.setattr(sp, "total_ram_gb", lambda: None)
    assert sp.fullres_ram_warning() is None


def test_total_ram_gb_returns_positive():
    # On the test host this should resolve via psutil or an OS fallback.
    ram = sp.total_ram_gb()
    assert ram is None or ram > 0


# --- analyze_dataset local default resolution (no real seg) ------------------

def test_analyze_local_defaults_to_fullres(monkeypatch):
    from spine_hu_tool.app import analysis

    captured = {}

    def _fake_segment(volume, work_dir, name, *, fast, local, seg_url=None,
                      api_key=None, progress=None):
        captured["fast"] = fast
        captured["local"] = local
        return np.zeros((2, 2, 2), dtype=np.int16)

    monkeypatch.setattr(analysis, "segment", _fake_segment)
    monkeypatch.setattr(analysis, "load_series",
                        lambda files, metadata=None: Volume(
                            hu=np.zeros((4, 4, 4), np.int16), spacing=(1.0, 1.0, 1.0)))
    monkeypatch.setattr(analysis.ReviewState, "from_volume",
                        classmethod(lambda cls, *a, **k: "STATE"))

    class _Series:
        series_uid = "1.2.3"
        study_uid = "1.2.3.4"
        n_files = 4
        description = kernel = manufacturer_model = ""
        slice_thickness = kvp = None
        files = ["a"]

    analysis.analyze_dataset("/folder", series=_Series(), local=True, scout=False)
    assert captured["local"] is True
    assert captured["fast"] is False  # full-res default for local
