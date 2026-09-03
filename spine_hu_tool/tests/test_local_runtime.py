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
    weights = tmp_path / "totalseg-weights"
    weights.mkdir()

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("SPINE_HU_BUNDLED_ENV", raising=False)
    monkeypatch.delenv("SPINE_HU_BUNDLED_WEIGHTS", raising=False)

    assert ls.bundled_env_dir() == str(envdir)
    assert ls.bundled_weights_dir() == str(weights)
    assert ls.weights_home() == str(weights)


def test_bundled_env_overrides(monkeypatch, tmp_path):
    d = tmp_path / "custom-env"
    d.mkdir()
    monkeypatch.setenv("SPINE_HU_BUNDLED_ENV", str(d))
    assert ls.bundled_env_dir() == str(d)
    monkeypatch.setenv("SPINE_HU_BUNDLED_ENV", str(tmp_path / "does-not-exist"))
    assert ls.bundled_env_dir() is None


def _make_fake_python(path):
    """Create a fake interpreter file (executable on POSIX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_find_runtime_python_standalone_layout(tmp_path):
    # uv installs python-build-standalone at <env>/cpython-<ver>-<platform>/
    envdir = tmp_path / "localseg-env"
    if sys.platform.startswith("win"):
        py = _make_fake_python(
            envdir / "cpython-3.11.9-windows-x86_64-none" / "python.exe")
    else:
        py = _make_fake_python(
            envdir / "cpython-3.11.9-macos-aarch64-none" / "bin" / "python3.11")
    assert ls.find_runtime_python(str(envdir)) == py


def test_find_runtime_python_venv_layout(tmp_path):
    # legacy venv layout (the managed env)
    envdir = tmp_path / "env"
    if sys.platform.startswith("win"):
        py = _make_fake_python(envdir / "Scripts" / "python.exe")
    else:
        py = _make_fake_python(envdir / "bin" / "python3")
    assert ls.find_runtime_python(str(envdir)) == py


def test_find_runtime_python_missing(tmp_path):
    assert ls.find_runtime_python(str(tmp_path)) is None
    assert ls.find_runtime_python(None) is None


def test_runtime_candidates_order_and_availability(monkeypatch):
    monkeypatch.setattr(ls, "bundled_python", lambda: "/bundle/py")
    monkeypatch.setattr(ls, "managed_python", lambda: "/managed/py")
    cands = ls.runtime_candidates()
    assert cands[0] == ("bundled", "/bundle/py")
    assert cands[1] == ("managed", "/managed/py")
    assert tr.local_seg_available() is True


def test_no_runtime_available(monkeypatch):
    import importlib.util
    monkeypatch.setattr(ls, "bundled_python", lambda: None)
    monkeypatch.setattr(ls, "managed_python", lambda: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _n: None)
    assert ls.runtime_candidates() == []
    assert tr._ts_python() is None
    assert tr.local_seg_available() is False


# --- runtime health check (verify_runtime) -----------------------------------

def test_verify_runtime_missing_interpreter():
    ok, reason = ls.verify_runtime("/no/such/python", use_cache=False)
    assert not ok and "not found" in reason


def test_verify_runtime_broken_runtime_captures_reason(tmp_path):
    # A runtime whose imports fail: health check must fail and carry the
    # import error as the reason. (A stub interpreter keeps this deterministic
    # regardless of whether the test venv has torch installed.)
    import pytest
    if sys.platform.startswith("win"):
        pytest.skip("POSIX stub interpreter")
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n"
                    "echo \"ModuleNotFoundError: No module named 'torch'\" >&2\n"
                    "exit 1\n")
    fake.chmod(0o755)
    ok, reason = ls.verify_runtime(str(fake), use_cache=False)
    assert not ok
    assert "ModuleNotFoundError" in reason


def test_verify_runtime_caches(monkeypatch):
    calls = {"n": 0}

    class _R:
        returncode = 0
        stdout = stderr = ""

    def _fake_run(*a, **k):
        calls["n"] += 1
        return _R()

    ls.clear_verify_cache()
    monkeypatch.setattr(ls.subprocess, "run", _fake_run)
    monkeypatch.setattr(ls.os.path, "exists", lambda _p: True)
    assert ls.verify_runtime("/fake/py")[0] is True
    assert ls.verify_runtime("/fake/py")[0] is True
    assert calls["n"] == 1
    ls.clear_verify_cache()


def test_resolve_runtime_error_names_paths_and_overrides(monkeypatch):
    from spine_hu_tool.errors import UserFacingError
    ls.clear_verify_cache()
    monkeypatch.setattr(ls, "bundled_python", lambda: "/broken/bundled/py")
    monkeypatch.setattr(ls, "managed_python", lambda: None)
    monkeypatch.setattr(ls, "verify_runtime",
                        lambda py, **k: (False, "ImportError: no torch"))
    monkeypatch.setenv("SPINE_HU_BUNDLED_ENV", "/broken/bundled")
    import pytest
    with pytest.raises(UserFacingError) as ei:
        ls.resolve_runtime()
    err = ei.value
    assert err.kind == "local-runtime-broken"
    assert "/broken/bundled/py" in err.detail
    assert "SPINE_HU_BUNDLED_ENV" in err.detail       # override flagged
    assert "Repair" in err.remedy and "cloud" in err.remedy


def test_resolve_runtime_prefers_first_healthy(monkeypatch):
    ls.clear_verify_cache()
    monkeypatch.setattr(ls, "bundled_python", lambda: "/broken/bundled/py")
    monkeypatch.setattr(ls, "managed_python", lambda: "/healthy/managed/py")
    monkeypatch.setattr(
        ls, "verify_runtime",
        lambda py, **k: (py == "/healthy/managed/py", "boom"))
    py, source = ls.resolve_runtime()
    assert py == "/healthy/managed/py" and source == "managed"


# --- managed setup: pinned uv dirs + self-heal (the 0.1.3.x hotfix line) -----

def test_uv_dirs_pinned_inside_user_data_root(monkeypatch, tmp_path):
    # uv's default roaming-profile python dir breaks on 448-hardened Windows
    # machines; all uv state must live under our own local data root.
    monkeypatch.setattr(ls, "_user_data_root", lambda: str(tmp_path))
    dirs = ls._uv_dirs()
    for key in ("UV_DATA_DIR", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"):
        assert dirs[key].startswith(str(tmp_path)), key


def test_uv_env_is_sanitized_and_pinned(monkeypatch, tmp_path):
    monkeypatch.setattr(ls, "_user_data_root", lambda: str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", "/frozen/leak")
    env = ls._uv_env()
    assert "PYTHONHOME" not in env                    # child_env sanitization
    assert env["UV_PYTHON_INSTALL_DIR"].startswith(str(tmp_path))


def test_find_installed_python_uses_versioned_dir_not_alias(monkeypatch, tmp_path):
    """Resolve the fully-versioned cpython-3.11.x dir by name; the bare
    minor-version alias (a junction/symlink on real installs -- exactly what
    hardened machines refuse to traverse) must never match."""
    monkeypatch.setattr(ls, "_user_data_root", lambda: str(tmp_path))
    install_dir = ls._uv_dirs()["UV_PYTHON_INSTALL_DIR"]
    versioned = os.path.join(install_dir, "cpython-3.11.9-plat")
    alias = os.path.join(install_dir, "cpython-3.11-plat")
    for d in (versioned, alias):
        sub = d if ls._IS_WIN else os.path.join(d, "bin")
        os.makedirs(sub, exist_ok=True)
        py = os.path.join(sub, "python.exe" if ls._IS_WIN else "python3")
        with open(py, "w") as f:
            f.write("")
    assert ls._find_installed_python() == (
        os.path.join(versioned, "python.exe") if ls._IS_WIN
        else os.path.join(versioned, "bin", "python3"))


def test_find_installed_python_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ls, "_user_data_root", lambda: str(tmp_path))
    assert ls._find_installed_python() is None


def test_setup_rebuilds_broken_env_from_scratch(monkeypatch, tmp_path):
    """A present-but-unhealthy env must be deleted before reprovisioning:
    running pip against its orphaned interpreter would fail with the same
    opaque errors the user is trying to escape."""
    env_dir = tmp_path / "localseg-env"
    env_dir.mkdir()
    (env_dir / "stale-marker").write_text("broken leftover")
    monkeypatch.setattr(ls, "managed_env_dir", lambda: str(env_dir))
    monkeypatch.setattr(ls, "is_ready", lambda: False)
    calls = {}

    def fake_provision(env_dir_arg, **kw):
        calls["env_dir"] = env_dir_arg
        calls["existed_at_provision"] = os.path.isdir(env_dir_arg)
        return "/fresh/py"

    monkeypatch.setattr(ls, "provision_env", fake_provision)
    monkeypatch.setattr(ls, "verify_runtime", lambda py, **k: (True, ""))
    assert ls.setup_local_seg() == "/fresh/py"
    assert calls["env_dir"] == str(env_dir)
    assert calls["existed_at_provision"] is False     # rebuilt from scratch


# --- run_segmentation builds a sanitized env + python -c command, never launches

def test_run_segmentation_env_and_command(monkeypatch, tmp_path):
    """run_segmentation must build a sanitized env with TOTALSEG_HOME_DIR and a
    python -c command (NEVER a console-script trampoline) -- captured via a fake
    subprocess so nothing is executed."""
    monkeypatch.setattr(ls, "resolve_runtime", lambda: ("/fake/env/python", "bundled"))
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
    cmd = captured["cmd"]
    assert "--fast" in cmd
    # invoked through the runtime's python -c entry, immune to trampolines
    assert cmd[0] == "/fake/env/python"
    assert cmd[1] == "-c" and "TotalSegmentator" in cmd[2]
    assert not any(str(c).endswith(("TotalSegmentator", "TotalSegmentator.exe"))
                   for c in cmd)


def test_seg_failure_error_flags_oom(tmp_path):
    from spine_hu_tool.errors import UserFacingError
    err = tr._seg_failure_error(["python", "-c", "x"], -9, "worker killed",
                                str(tmp_path / "seg.log"))
    assert isinstance(err, UserFacingError)
    assert err.kind == "seg-oom"
    assert "memory" in err.message.lower()

    err = tr._seg_failure_error(["python", "-c", "x"], 1, "RuntimeError: boom",
                                str(tmp_path / "seg.log"))
    assert err.kind == "seg-crashed"
    assert "cloud" in err.remedy.lower()


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

    class _State:
        case = {}
        volume = None

    monkeypatch.setattr(analysis, "segment", _fake_segment)
    monkeypatch.setattr(analysis, "load_series",
                        lambda files, metadata=None: Volume(
                            hu=np.zeros((4, 4, 4), np.int16), spacing=(1.0, 1.0, 1.0)))
    monkeypatch.setattr(analysis.ReviewState, "from_volume",
                        classmethod(lambda cls, *a, **k: _State()))

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
