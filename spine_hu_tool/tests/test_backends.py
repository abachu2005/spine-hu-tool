"""Remote segmentation backend: dispatch + signed-URL/polling flow (mocked).

No network/server needed. We monkeypatch requests.post/put/get to emulate the
robust async flow:
  POST /upload-url           -> {object_name, url, content_type}
  PUT  <signed url>          -> 200 (volume "uploaded" to GCS)
  POST /segment-async        -> {job_id}
  GET  /segment-status/{id}  -> {status: done, result_url: <signed get>}
  GET  <signed get>          -> streams a real .nii.gz mask back
so the upload/poll/download/cache path is tested deterministically.
"""
from __future__ import annotations
import os
import numpy as np

from spine_hu_tool.core import Volume
from spine_hu_tool.io.nifti_io import save_mask_nifti
from spine_hu_tool.segmentation import backends


def _volume():
    hu = np.zeros((12, 12, 12), dtype=np.float32)
    return Volume(hu=hu, spacing=(1.0, 1.0, 2.0))


class _FakeResp:
    def __init__(self, path=None, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self._bytes = open(path, "rb").read() if path else b""

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._bytes), chunk_size):
            yield self._bytes[i:i + chunk_size]


def test_resolve_seg_url(monkeypatch):
    # explicit arg and env override always win
    monkeypatch.delenv("SPINE_HU_SEG_URL", raising=False)
    assert backends.resolve_seg_url("http://x/") == "http://x"
    monkeypatch.setenv("SPINE_HU_SEG_URL", "http://env:8080/")
    assert backends.resolve_seg_url(None) == "http://env:8080"


def test_resolve_seg_url_defaults_to_cloud(monkeypatch):
    # Cloud is the default backend: with no override and no creds file, the
    # baked-in deployed URL is used (never None -> never silently local).
    monkeypatch.delenv("SPINE_HU_SEG_URL", raising=False)
    monkeypatch.setattr(backends, "_from_creds_file", lambda key: None)
    assert backends.resolve_seg_url(None) == backends.DEFAULT_SEG_URL


def test_segment_local_when_no_url(monkeypatch, tmp_path):
    called = {}

    def fake_local(volume, work_dir, name, fast, force):
        called["hit"] = (fast, force)
        return np.ones((2, 2, 2), dtype=np.int16)

    monkeypatch.delenv("SPINE_HU_SEG_URL", raising=False)
    # explicitly opt out of cloud (no URL anywhere) -> local fallback
    monkeypatch.setattr(backends, "_from_creds_file", lambda key: None)
    monkeypatch.setattr(backends, "DEFAULT_SEG_URL", "")
    monkeypatch.setattr(backends, "run_segmentation", fake_local)
    out = backends.segment(_volume(), str(tmp_path), "case", fast=True)
    assert called["hit"] == (True, False)
    assert out.shape == (2, 2, 2)


def _install_fake_flow(monkeypatch, truth_path, *, start_status=200,
                       status_payloads=None, status_error=None):
    """Wire requests.post/put/get to emulate the async signed-URL flow.

    status_payloads: list of {status:...} dicts returned by successive
    /segment-status polls (default: a single 'done').
    """
    import time
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    cap = {"posts": [], "gets": [], "put": None}
    polls = list(status_payloads or [{"status": "done",
                                      "result_url": "https://gcs.example/signed-get"}])

    def fake_post(url, headers=None, json=None, timeout=None, stream=None, **kw):
        cap["posts"].append({"url": url, "headers": headers, "json": json})
        if url.endswith("/upload-url"):
            return _FakeResp(payload={"object_name": "uploads/abc.nii.gz",
                                      "url": "https://gcs.example/signed-put",
                                      "content_type": "application/octet-stream"})
        if url.endswith("/segment-async"):
            if start_status != 200:
                return _FakeResp(status_code=start_status, text="nope")
            return _FakeResp(payload={"job_id": "job123"})
        raise AssertionError(f"unexpected POST {url}")

    def fake_put(url, data=None, headers=None, timeout=None, **kw):
        cap["put"] = {"url": url, "headers": headers}
        return _FakeResp(status_code=200)

    def fake_get(url, headers=None, timeout=None, stream=None, **kw):
        cap["gets"].append(url)
        if "/segment-status/" in url:
            return _FakeResp(payload=polls.pop(0))
        if url == "https://gcs.example/signed-get":
            return _FakeResp(path=truth_path)
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(backends, "run_segmentation",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not run locally")))
    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "put", fake_put)
    monkeypatch.setattr(requests, "get", fake_get)
    return cap


def test_segment_remote_roundtrip(monkeypatch, tmp_path):
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:6, 3:6, 3:6] = 28        # e.g. vertebrae_T8 label id
    truth_path = str(tmp_path / "truth_seg.nii.gz")
    save_mask_nifti(mask, (1.0, 1.0, 2.0), truth_path)

    # first poll still running, second poll done -> exercises the loop
    cap = _install_fake_flow(monkeypatch, truth_path, status_payloads=[
        {"status": "running"},
        {"status": "done", "result_url": "https://gcs.example/signed-get"},
    ])
    seg = backends.segment(_volume(), str(tmp_path), "case", fast=True,
                           seg_url="http://server:8080", api_key="secret")

    post_urls = [p["url"] for p in cap["posts"]]
    assert post_urls == ["http://server:8080/upload-url",
                         "http://server:8080/segment-async"]
    # auth header sent to our endpoints, and the big PUT went to the signed URL
    assert all(p["headers"]["Authorization"] == "Bearer secret" for p in cap["posts"])
    assert cap["put"]["url"] == "https://gcs.example/signed-put"
    # start request references the uploaded object + fast flag
    assert cap["posts"][1]["json"] == {"object_name": "uploads/abc.nii.gz", "fast": True}
    # polled status then downloaded the mask from the signed GET url
    assert any("/segment-status/job123" in g for g in cap["gets"])
    assert cap["gets"][-1] == "https://gcs.example/signed-get"
    # mask written to cache and round-trips
    assert os.path.exists(tmp_path / "case_seg.nii.gz")
    assert int(seg.max()) == 28 and seg.shape == (12, 12, 12)


def test_segment_remote_uses_cache(monkeypatch, tmp_path):
    # pre-seed the cache; remote must NOT be called
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    save_mask_nifti(mask, (1.0, 1.0, 1.0), str(tmp_path / "case_seg.nii.gz"))

    import requests
    boom = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("cache should prevent network call"))
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "put", boom)
    monkeypatch.setattr(requests, "get", boom)
    seg = backends.segment(_volume(), str(tmp_path), "case",
                           seg_url="http://server:8080")
    assert seg.shape == (4, 4, 4)


def test_segment_remote_error_raises(monkeypatch, tmp_path):
    # server reports the job failed -> client raises with the error text
    _install_fake_flow(monkeypatch, truth_path=None, status_payloads=[
        {"status": "error", "error": "boom"},
    ])
    try:
        backends.segment(_volume(), str(tmp_path), "case",
                         seg_url="http://server:8080")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "boom" in str(e)


def test_adaptive_timeout_scales_with_slices():
    # A big full-res scan gets a longer deadline than a small one (up to a cap),
    # and fast mode always gets a shorter deadline than full-res for the same scan.
    small = Volume(hu=np.zeros((8, 8, 10), dtype=np.float32), spacing=(1., 1., 2.))
    big = Volume(hu=np.zeros((8, 8, 400), dtype=np.float32), spacing=(1., 1., 2.))
    assert backends._adaptive_timeout(small, fast=False) == 1200.0   # floor
    assert backends._adaptive_timeout(big, fast=False) == 1800.0     # ceiling
    assert backends._adaptive_timeout(big, fast=False) > \
        backends._adaptive_timeout(big, fast=True)


def test_segment_falls_back_to_fast_on_timeout(monkeypatch, tmp_path):
    # A full-resolution remote job that exceeds its deadline must NOT hard-fail:
    # it transparently retries in fast (3 mm) mode and still returns a mask.
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:6, 3:6, 3:6] = 28
    truth_path = str(tmp_path / "truth_seg.nii.gz")
    save_mask_nifti(mask, (1.0, 1.0, 2.0), truth_path)

    # the (only) status poll reports success; the first (full-res) attempt never
    # reaches it because we force its deadline to expire immediately.
    cap = _install_fake_flow(monkeypatch, truth_path, status_payloads=[
        {"status": "done", "result_url": "https://gcs.example/signed-get"},
    ])
    msgs = []
    seg = backends.segment(_volume(), str(tmp_path), "case", fast=False,
                           seg_url="http://server:8080", timeout=0.0,
                           progress=lambda m, _f: msgs.append(m))

    # two segmentation jobs were started: full-res (timed out) then fast fallback
    starts = [p for p in cap["posts"] if p["url"].endswith("/segment-async")]
    assert [s["json"]["fast"] for s in starts] == [False, True]
    assert any("fast (3 mm) mode" in m for m in msgs)
    assert int(seg.max()) == 28 and seg.shape == (12, 12, 12)


def test_segment_no_fallback_when_already_fast(monkeypatch, tmp_path):
    # a fast job that times out has nothing lighter to fall back to -> it raises.
    _install_fake_flow(monkeypatch, truth_path=None, status_payloads=[])
    try:
        backends.segment(_volume(), str(tmp_path), "case", fast=True,
                         seg_url="http://server:8080", timeout=0.0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "timed out" in str(e)
