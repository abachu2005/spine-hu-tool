"""Server smoke test (skipped unless the server stack is installed).

Verifies the FastAPI app builds, /health responds, and auth gates /segment.
The heavy TotalSegmentator call is monkeypatched so no model runs here.
"""
from __future__ import annotations
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


def _client(monkeypatch, api_key=None):
    from spine_hu_tool.server import app as server_app
    # Always pin API_KEY (default None = auth off) so tests are deterministic
    # regardless of any SPINE_HU_API_KEY in the environment.
    monkeypatch.setattr(server_app, "API_KEY", api_key)
    return TestClient(server_app.app), server_app


def test_health(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_segment_requires_auth_when_key_set(monkeypatch):
    client, _ = _client(monkeypatch, api_key="secret")
    r = client.post("/segment", files={"file": ("x.nii.gz", b"data", "application/gzip")})
    assert r.status_code == 401


def _fake_seg(server_app, monkeypatch):
    """Stub the loader + segmentation so no real model/IO runs."""
    monkeypatch.setattr(server_app, "load_volume_from_nifti", lambda p: object())

    def fake_run(volume, work_dir, name, fast=True, force=False):
        from spine_hu_tool.io.nifti_io import save_mask_nifti
        import os
        m = np.zeros((3, 3, 3), dtype=np.uint8)
        save_mask_nifti(m, (1.0, 1.0, 1.0), os.path.join(work_dir, f"{name}_seg.nii.gz"))
        return m

    monkeypatch.setattr(server_app, "run_segmentation", fake_run)


def test_segment_runs_and_returns_mask(monkeypatch, tmp_path):
    client, server_app = _client(monkeypatch)
    _fake_seg(server_app, monkeypatch)
    r = client.post("/segment",
                    files={"file": ("x.nii.gz", b"data", "application/gzip")},
                    data={"fast": "true"})
    assert r.status_code == 200
    assert len(r.content) > 0


def test_upload_url(monkeypatch):
    client, server_app = _client(monkeypatch)
    monkeypatch.setattr(server_app, "_signed_put_url",
                        lambda obj, ct: f"https://signed/{obj}?ct={ct}")
    r = client.post("/upload-url")
    assert r.status_code == 200
    body = r.json()
    assert body["object_name"].startswith("uploads/")
    assert body["url"].startswith("https://signed/uploads/")
    assert body["content_type"] == "application/octet-stream"


def test_segment_gcs_downloads_and_returns_mask(monkeypatch):
    client, server_app = _client(monkeypatch)
    _fake_seg(server_app, monkeypatch)

    deleted = {"called": False}

    class _FakeBlob:
        def __init__(self, name): self.name = name
        def download_to_filename(self, path):
            open(path, "wb").write(b"fake nifti bytes")
        def delete(self): deleted["called"] = True

    class _FakeBucket:
        def blob(self, name): return _FakeBlob(name)

    monkeypatch.setattr(server_app, "_bucket", lambda: _FakeBucket())

    r = client.post("/segment-gcs", json={"object_name": "uploads/x.nii.gz", "fast": True})
    assert r.status_code == 200
    assert len(r.content) > 0
    assert deleted["called"], "uploaded object should be deleted after segmentation"


def test_segment_async_poll_and_download(monkeypatch):
    import time
    client, server_app = _client(monkeypatch)
    _fake_seg(server_app, monkeypatch)

    state = {"uploaded": None, "deleted": False}

    class _FakeBlob:
        def __init__(self, name): self.name = name
        def download_to_filename(self, path):
            open(path, "wb").write(b"fake nifti bytes")
        def upload_from_filename(self, path):
            state["uploaded"] = self.name
        def delete(self): state["deleted"] = True

    class _FakeBucket:
        def blob(self, name): return _FakeBlob(name)

    monkeypatch.setattr(server_app, "_bucket", lambda: _FakeBucket())
    monkeypatch.setattr(server_app, "_signed_url",
                        lambda obj, method, content_type=None: f"https://signed/{obj}")

    r = client.post("/segment-async", json={"object_name": "uploads/x.nii.gz", "fast": True})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # poll until the background worker finishes
    final = None
    for _ in range(50):
        s = client.get(f"/segment-status/{job_id}")
        assert s.status_code == 200
        final = s.json()
        if final["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    assert final["status"] == "done", final
    assert final["result_url"] == "https://signed/results/%s.nii.gz" % job_id
    assert state["uploaded"] == f"results/{job_id}.nii.gz"
    assert state["deleted"], "input upload should be deleted after segmentation"


def test_segment_status_unknown_job(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/segment-status/does-not-exist")
    assert r.status_code == 404


# ---- /runs/upload-url (run archival) ----------------------------------------

def test_runs_upload_url_mints_scoped_object(monkeypatch):
    client, server_app = _client(monkeypatch)
    monkeypatch.setattr(server_app, "_signed_put_url",
                        lambda obj, ct: f"https://signed/{obj}?ct={ct}")
    r = client.post("/runs/upload-url",
                    json={"run_id": "20260101T000000Z-abcd1234",
                          "filename": "manifest.json"})
    assert r.status_code == 200
    body = r.json()
    import re
    assert re.fullmatch(
        r"runs/\d{4}-\d{2}/20260101T000000Z-abcd1234/manifest\.json",
        body["object_name"]), body["object_name"]
    assert body["url"].startswith("https://signed/runs/")


def test_runs_upload_url_requires_auth_when_key_set(monkeypatch):
    client, server_app = _client(monkeypatch, api_key="secret")
    monkeypatch.setattr(server_app, "_signed_put_url",
                        lambda obj, ct: f"https://signed/{obj}")
    r = client.post("/runs/upload-url",
                    json={"run_id": "r", "filename": "manifest.json"})
    assert r.status_code == 401
    r = client.post("/runs/upload-url",
                    headers={"Authorization": "Bearer secret"},
                    json={"run_id": "r1", "filename": "f.json"})
    assert r.status_code == 200


def test_runs_upload_url_sanitizes_traversal(monkeypatch):
    client, server_app = _client(monkeypatch)
    monkeypatch.setattr(server_app, "_signed_put_url",
                        lambda obj, ct: f"https://signed/{obj}")
    r = client.post("/runs/upload-url",
                    json={"run_id": "../../etc", "filename": "../../passwd"})
    assert r.status_code == 200
    obj = r.json()["object_name"]
    assert ".." not in obj
    # both components collapsed to safe names inside the runs/ prefix
    assert obj.startswith("runs/") and obj.endswith("/passwd")

    r = client.post("/runs/upload-url",
                    json={"run_id": "a/b\\c", "filename": "x y!.json"})
    assert r.status_code == 200
    obj = r.json()["object_name"]
    assert " " not in obj and "\\" not in obj
    assert obj.count("/") == 3          # runs/<month>/<run_id>/<filename>


def test_runs_upload_url_rejects_empty_components(monkeypatch):
    client, server_app = _client(monkeypatch)
    monkeypatch.setattr(server_app, "_signed_put_url",
                        lambda obj, ct: f"https://signed/{obj}")
    r = client.post("/runs/upload-url", json={"run_id": "..", "filename": "f"})
    assert r.status_code == 400
    r = client.post("/runs/upload-url", json={"run_id": "r", "filename": "..."})
    assert r.status_code == 400
