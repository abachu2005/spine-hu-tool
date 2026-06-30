"""Segmentation backend dispatch: local subprocess vs. remote service.

The app calls :func:`segment`; whether the heavy TotalSegmentator step runs on
this machine or on a Cloud Run service is a deployment detail decided by a URL.
Either way the result is cached on disk (keyed by `name`) so re-opening a
dataset is instant and never re-segments.

Resolution order for the remote endpoint:
  1. explicit `seg_url` argument
  2. env var SPINE_HU_SEG_URL
  3. (none) -> run locally
"""
from __future__ import annotations
import os
from typing import Callable, Optional

import numpy as np

from ..core import Volume
from ..io.nifti_io import save_volume_nifti
from .totalseg_runner import run_segmentation, load_segmentation

ProgressCb = Optional[Callable[[str, float], None]]

# Cloud Run is the DEFAULT backend: the heavy TotalSegmentator step is never run
# on the clinician's machine unless the URL is explicitly cleared. The endpoint
# and key are resolved from (1) the explicit argument, (2) env vars, (3) the
# local deploy credentials file, (4) the baked-in deployed service URL.
DEFAULT_SEG_URL = "https://spine-hu-seg-980966741284.us-central1.run.app"

# Shared pilot key baked into the client so downloaded installers authenticate
# out of the box (the server is gated only to keep random internet traffic from
# running up cost). This is intentionally a low-value, rotatable shared secret
# for a private pilot -- NOT per-user auth. Rotate by updating the Cloud Run
# SPINE_HU_API_KEY env var and this constant. Override with the SPINE_HU_API_KEY
# env var or work/cloud_run_credentials.txt during development.
DEFAULT_API_KEY = "AQPpyHBhOqjvDGOIQwyClBbkXID1JKNw"

_CREDS_FILE = os.path.join(os.path.dirname(__file__), "..", "..",
                           "work", "cloud_run_credentials.txt")


def _from_creds_file(key: str) -> Optional[str]:
    # The dev credentials file is a developer convenience only. In a packaged
    # (frozen) app it does not exist, so skip it entirely and rely on the env
    # var + baked-in DEFAULT_SEG_URL. The read is guarded regardless.
    import sys
    if getattr(sys, "frozen", False):
        return None
    try:
        with open(_CREDS_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip() or None
    except OSError:
        return None
    return None


def resolve_seg_url(seg_url: Optional[str]) -> Optional[str]:
    url = (seg_url or os.environ.get("SPINE_HU_SEG_URL")
           or _from_creds_file("SPINE_HU_SEG_URL") or DEFAULT_SEG_URL)
    return url.rstrip("/") if url else None


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    return (api_key or os.environ.get("SPINE_HU_API_KEY")
            or _from_creds_file("SPINE_HU_API_KEY") or DEFAULT_API_KEY)


def segment(volume: Volume, work_dir: str, name: str, *,
            fast: bool = True, force: bool = False, local: bool = False,
            seg_url: Optional[str] = None, api_key: Optional[str] = None,
            timeout: float = 1200.0, progress: ProgressCb = None):
    """Return the multilabel segmentation array, running locally or remotely.

    Remote and local share the same on-disk cache (`{name}_seg.nii.gz` in
    `work_dir`), so switching backends never forces a re-run of cached cases.

    `local=True` forces on-machine segmentation regardless of any configured
    server URL (env / baked-in default). Otherwise the remote endpoint is
    resolved and used; only when no URL resolves does it fall back to local.
    """
    url = None if local else resolve_seg_url(seg_url)
    if url is None:
        return run_segmentation(volume, work_dir, name, fast=fast, force=force)
    return _segment_remote(volume, work_dir, name, url, api_key,
                           fast=fast, force=force, timeout=timeout,
                           progress=progress)


def _segment_remote(volume: Volume, work_dir: str, name: str, url: str,
                    api_key: Optional[str], *, fast: bool, force: bool,
                    timeout: float, progress: ProgressCb,
                    poll_interval: float = 5.0):
    """Signed-URL + polling flow (robust against long segmentations):

      1. POST /upload-url            -> {object_name, url, content_type}
      2. PUT  <signed url>           -> volume bytes go directly to GCS
      3. POST /segment-async         -> {job_id}   (returns immediately)
      4. GET  /segment-status/{id}   -> poll until done; get a signed mask URL
      5. GET  <signed mask url>      -> download the (small) mask from GCS

    Every request is short, so nothing hangs on a minutes-long open connection.
    """
    import time
    import requests  # local import so the client dep is only needed for remote

    os.makedirs(work_dir, exist_ok=True)
    seg_path = os.path.join(work_dir, f"{name}_seg.nii.gz")
    if os.path.exists(seg_path) and not force:
        return load_segmentation(seg_path)

    # int16 = lossless for CT HU and ~half the upload size
    in_path = os.path.join(work_dir, f"{name}_upload.nii.gz")
    save_volume_nifti(volume, in_path, dtype=np.int16)

    api_key = resolve_api_key(api_key)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _report(msg):
        if progress:
            progress(msg, -1.0)

    import random

    def _with_retry(label, fn, attempts=4):
        """Run an HTTP step, retrying transient (5xx / network) failures.

        Cloud Run and GCS occasionally return 5xx (e.g. a 504 InternalError on
        a signed-URL PUT); these are explicitly "please try again" and succeed
        on retry. 4xx are caller errors and fail fast.
        """
        last_err = None
        for i in range(attempts):
            try:
                resp = fn()
            except requests.RequestException as exc:
                last_err = RuntimeError(f"{label} failed (network): {exc}")
            else:
                if resp.status_code in (200, 201):
                    return resp
                if resp.status_code < 500:
                    raise RuntimeError(
                        f"{label} failed ({resp.status_code}): {resp.text[:300]}")
                last_err = RuntimeError(
                    f"{label} failed ({resp.status_code}): {resp.text[:200]}")
            if i < attempts - 1:
                wait = min(2 ** i, 8) + random.random()
                _report(f"{label}: transient cloud error, retrying in {wait:.0f}s...")
                time.sleep(wait)
        raise last_err or RuntimeError(f"{label} failed")

    # 1. mint a signed upload URL
    _report("Requesting upload URL...")
    r = _with_retry("upload-url",
                    lambda: requests.post(f"{url}/upload-url",
                                          headers=headers, timeout=60))
    up = r.json()

    # 2. upload the volume directly to GCS via the signed URL. Reopen the file
    # on each attempt so a retry re-streams from the start.
    _report("Uploading scan to cloud storage...")

    def _put_volume():
        with open(in_path, "rb") as fh:
            return requests.put(up["url"], data=fh,
                                headers={"Content-Type": up["content_type"]},
                                timeout=600)

    _with_retry("GCS upload", _put_volume)

    # 3. start segmentation; the server returns a job id immediately
    _report("Starting segmentation on the cloud service...")
    sr = _with_retry("start segmentation",
                     lambda: requests.post(
                         f"{url}/segment-async", headers=headers,
                         json={"object_name": up["object_name"], "fast": fast},
                         timeout=60))
    job_id = sr.json()["job_id"]

    # 4. poll until the job finishes (short requests -> no hung connections)
    _report("Segmenting on the cloud service (this can take a few minutes)...")
    deadline = time.time() + timeout
    result_url = None
    while time.time() < deadline:
        time.sleep(poll_interval)
        try:
            st = requests.get(f"{url}/segment-status/{job_id}",
                              headers=headers, timeout=30)
        except requests.RequestException:
            continue   # transient network blip; keep polling
        if st.status_code != 200:
            raise RuntimeError(
                f"Status check failed ({st.status_code}): {st.text[:300]}")
        info = st.json()
        if info["status"] == "done":
            result_url = info["result_url"]
            break
        if info["status"] == "error":
            raise RuntimeError(f"Remote segmentation failed: {info.get('error')}")
    if result_url is None:
        raise RuntimeError("Remote segmentation timed out")

    # 5. download the mask straight from GCS
    _report("Downloading segmentation mask...")
    dl = requests.get(result_url, timeout=600, stream=True)
    if dl.status_code != 200:
        raise RuntimeError(f"Mask download failed ({dl.status_code}): {dl.text[:300]}")
    with open(seg_path, "wb") as out:
        for chunk in dl.iter_content(chunk_size=1 << 20):
            if chunk:
                out.write(chunk)

    try:
        os.remove(in_path)
    except OSError:
        pass
    return load_segmentation(seg_path)
