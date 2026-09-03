"""Cloud run-archival tests: expected files against a mocked server, the
SPINE_HU_NO_ARCHIVE kill switch, and the never-break-an-analysis guarantee.
No test touches the network: `requests` is monkeypatched throughout."""
from __future__ import annotations
import json
import re

import numpy as np
import pytest

from spine_hu_tool.export import cloud_archive as ca


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeRequests:
    """Captures the /runs/upload-url mints and the PUT bodies."""

    def __init__(self):
        self.minted = []          # [(run_id, filename)]
        self.put_bodies = {}      # filename -> bytes

    def post(self, url, headers=None, json=None, timeout=None):
        assert url.endswith("/runs/upload-url")
        self.minted.append((json["run_id"], json["filename"]))
        return _Resp({"object_name": f"runs/x/{json['run_id']}/{json['filename']}",
                      "url": f"https://signed/{json['filename']}",
                      "content_type": "application/octet-stream"})

    def put(self, url, data=None, headers=None, timeout=None):
        name = url.rsplit("/", 1)[-1]
        if hasattr(data, "read"):
            data = data.read()
        self.put_bodies[name] = data
        return _Resp({})


@pytest.fixture
def fake_requests(monkeypatch):
    import requests
    fake = _FakeRequests()
    monkeypatch.setenv("SPINE_HU_NO_ARCHIVE", "0")
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr(requests, "put", fake.put)
    return fake


def test_run_id_is_sortable_and_unique():
    a, b = ca.new_run_id(), ca.new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", a)
    assert a != b


def test_kill_switch_disables_everything(monkeypatch):
    monkeypatch.setenv("SPINE_HU_NO_ARCHIVE", "1")
    assert ca.archiving_disabled()
    called = {"n": 0}
    arch = ca.RunArchiver()
    monkeypatch.setattr(arch, "_upload_all",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    arch.submit({"x": 1})
    assert called["n"] == 0


def test_upload_all_sends_manifest_and_files(fake_requests, tmp_path):
    f1 = tmp_path / "seg.nii.gz"; f1.write_bytes(b"mask-bytes")
    f2 = tmp_path / "totalseg.log"; f2.write_text("log line")
    arch = ca.RunArchiver(run_id="RUN1")
    # call synchronously (submit() would use a daemon thread)
    arch._upload_all({"run_id": "RUN1", "backend": "local"},
                     {"seg.nii.gz": str(f1), "totalseg.log": str(f2),
                      "missing.bin": str(tmp_path / "nope")})

    minted_names = {n for _rid, n in fake_requests.minted}
    assert minted_names == {"seg.nii.gz", "totalseg.log", "manifest.json"}
    assert all(rid == "RUN1" for rid, _n in fake_requests.minted)
    assert fake_requests.put_bodies["seg.nii.gz"] == b"mask-bytes"
    manifest = json.loads(fake_requests.put_bodies["manifest.json"])
    assert manifest["backend"] == "local"
    # the manifest records which files actually made it (missing one skipped)
    assert sorted(manifest["archived_files"]) == ["seg.nii.gz", "totalseg.log"]


def test_upload_all_is_silent_offline(monkeypatch):
    import requests
    monkeypatch.setenv("SPINE_HU_NO_ARCHIVE", "0")

    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("offline")
    monkeypatch.setattr(requests, "post", _boom)
    arch = ca.RunArchiver(run_id="RUN2")
    arch._upload_all({"run_id": "RUN2"}, {})     # must not raise


def test_failure_hook_carries_user_facing_error(fake_requests, monkeypatch):
    from spine_hu_tool.errors import UserFacingError

    submitted = {}

    arch = ca.RunArchiver(run_id="RUNF")
    monkeypatch.setattr(
        arch, "submit",
        lambda manifest, files=None: submitted.update(manifest=manifest,
                                                      files=files))
    err = UserFacingError("T", "Local segmentation isn't working.",
                          remedy="Repair it.", detail="paths...",
                          kind="local-runtime-broken")
    ca.archive_analysis_failure(arch, err, backend="local",
                                mode="m", fast=False, folder="/data/case1")
    m = submitted["manifest"]
    assert m["status"] == "failed"
    assert m["error"]["kind"] == "local-runtime-broken"
    # the archived manifest shows exactly what the user saw
    assert m["error"]["remedy"] == "Repair it."
    assert "traceback" in m["error"]
    assert m["source_folder_name"] == "case1"


def test_hooks_never_raise_into_analysis(monkeypatch):
    # Even with a completely broken state object / archiver, the hooks swallow.
    monkeypatch.setenv("SPINE_HU_NO_ARCHIVE", "0")

    class _BrokenState:
        @property
        def case(self):
            raise RuntimeError("boom")

    arch = ca.RunArchiver(run_id="RUNX")
    monkeypatch.setattr(arch, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    ca.archive_analysis_success(arch, _BrokenState(), backend="local",
                                mode="m", fast=True, series=None,
                                work_dir="/nope", name="n")   # no raise
    ca.archive_analysis_failure(arch, RuntimeError("y"), backend="local",
                                mode="m", fast=True)          # no raise


def test_analyze_dataset_tags_archive_run_id(monkeypatch):
    """analyze_dataset stamps the run id into state.case and calls the success
    hook exactly once, with segmentation stubbed."""
    from spine_hu_tool.app import analysis
    from spine_hu_tool.core import Volume

    hooks = {"success": 0}

    class _State:
        def __init__(self):
            self.case = {}
            self.volume = None

    monkeypatch.setattr(
        analysis, "segment",
        lambda *a, **k: np.zeros((2, 2, 2), dtype=np.int16))
    monkeypatch.setattr(
        analysis, "load_series",
        lambda files, metadata=None: Volume(hu=np.zeros((4, 4, 4), np.int16),
                                            spacing=(1.0, 1.0, 1.0)))
    monkeypatch.setattr(analysis.ReviewState, "from_volume",
                        classmethod(lambda cls, *a, **k: _State()))
    monkeypatch.setattr(
        analysis, "archive_analysis_success",
        lambda archiver, state, **k: hooks.__setitem__("success",
                                                       hooks["success"] + 1))

    class _Series:
        series_uid = "1.2.3"
        study_uid = "1.2.3.4"
        n_files = 4
        description = kernel = manufacturer_model = ""
        slice_thickness = kvp = None
        files = ["a"]

    st = analysis.analyze_dataset("/folder", series=_Series(), scout=False)
    assert hooks["success"] == 1
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", st.case["archive_run_id"])


def test_analyze_dataset_archives_failures(monkeypatch):
    from spine_hu_tool.app import analysis
    from spine_hu_tool.core import Volume

    captured = {}

    def _fail_segment(*a, **k):
        raise RuntimeError("seg exploded")

    monkeypatch.setattr(analysis, "segment", _fail_segment)
    monkeypatch.setattr(
        analysis, "load_series",
        lambda files, metadata=None: Volume(hu=np.zeros((4, 4, 4), np.int16),
                                            spacing=(1.0, 1.0, 1.0)))
    monkeypatch.setattr(
        analysis, "archive_analysis_failure",
        lambda archiver, exc, **k: captured.update(exc=exc, **k))

    class _Series:
        series_uid = "1.2.3"
        study_uid = "1.2.3.4"
        n_files = 4
        description = kernel = manufacturer_model = ""
        slice_thickness = kvp = None
        files = ["a"]

    with pytest.raises(RuntimeError, match="seg exploded"):
        analysis.analyze_dataset("/folder", series=_Series(), scout=False)
    assert str(captured["exc"]) == "seg exploded"
    assert captured["backend"] in ("local", "cloud")
    assert captured["folder"] == "/folder"
