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
from ..errors import UserFacingError
from ..io.nifti_io import save_volume_nifti
from .totalseg_runner import run_segmentation, load_segmentation

ProgressCb = Optional[Callable[[str, float], None]]

# Download page linked from the "app outdated" dialog.
DOWNLOAD_PAGE_URL = "https://storage.googleapis.com/spine-hu-tool-downloads/index.html"


def _cloud_unreachable(detail: str, *, timed_out: bool = False) -> UserFacingError:
    return UserFacingError(
        title="Couldn't reach the segmentation service",
        message=("The cloud segmentation run took too long and timed out."
                 if timed_out else
                 "The segmentation service couldn't be reached."),
        remedy="Check the internet connection and try again; if you're on a "
               "hospital network, a firewall may be blocking it. The server "
               "URL is on the start screen.",
        detail=detail, kind="cloud-timeout" if timed_out else "cloud-unreachable")


def _cloud_rejected(status: int, detail: str) -> UserFacingError:
    if status in (401, 403):
        return UserFacingError(
            title="Segmentation service rejected the request",
            message="The segmentation service rejected the request -- this "
                    "usually means the app is outdated.",
            remedy="Please download the latest version from the download "
                   f"page: {DOWNLOAD_PAGE_URL}",
            detail=detail, kind="cloud-rejected")
    return UserFacingError(
        title="Segmentation service rejected the request",
        message=f"The segmentation service rejected the request "
                f"(HTTP {status}).",
        remedy="Try again; if this keeps happening, download the latest "
               f"version from {DOWNLOAD_PAGE_URL} or send us the log file.",
        detail=detail, kind="cloud-rejected")

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
DEFAULT_API_KEY = "REMOVED_CLOUD_API_KEY"

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


def _adaptive_timeout(volume: Volume, fast: bool) -> float:
    """Poll deadline (seconds) scaled to the scan size.

    Remote segmentation runs on CPU and, on a scale-to-zero Cloud Run instance,
    the background worker is CPU-throttled between the client's short status
    polls, so wall-clock time grows roughly linearly with the number of slices.
    A large full-resolution scan (e.g. a 250+ slice chest/abdomen CT) can exceed
    the old fixed 1200 s cap and hard-fail. Scale the deadline with slice count
    (with a floor and a sane ceiling) so realistic scans finish, and let the
    fast-mode fallback in :func:`segment` catch the extreme tail.
    """
    try:
        slices = int(volume.hu.shape[2]) if volume.hu.ndim == 3 else 0
    except Exception:
        slices = 0
    if fast:
        return float(min(1200.0, max(600.0, slices * 4.0)))
    # ~10 s/slice observed on a throttled CPU instance; add margin, cap so we
    # fall back to fast mode rather than making the user wait indefinitely.
    return float(min(1800.0, max(1200.0, slices * 12.0)))


def segment(volume: Volume, work_dir: str, name: str, *,
            fast: bool = True, force: bool = False, local: bool = False,
            seg_url: Optional[str] = None, api_key: Optional[str] = None,
            timeout: Optional[float] = None, progress: ProgressCb = None):
    """Return the multilabel segmentation array, running locally or remotely.

    Remote and local share the same on-disk cache (`{name}_seg.nii.gz` in
    `work_dir`), so switching backends never forces a re-run of cached cases.

    `local=True` forces on-machine segmentation regardless of any configured
    server URL (env / baked-in default). Otherwise the remote endpoint is
    resolved and used; only when no URL resolves does it fall back to local.

    Robustness: a full-resolution remote job that exceeds its (size-aware)
    deadline does not hard-fail. Because the fast (3 mm) mask is resampled to the
    full-resolution grid and leaves the deterministic ROI placement intact, we
    transparently retry once in fast mode so the physician still gets a
    reviewable result, surfacing a clear warning via `progress`.
    """
    url = None if local else resolve_seg_url(seg_url)
    if url is None:
        return run_segmentation(volume, work_dir, name, fast=fast, force=force)
    eff_timeout = timeout if timeout is not None else _adaptive_timeout(volume, fast)
    try:
        return _segment_remote(volume, work_dir, name, url, api_key,
                               fast=fast, force=force, timeout=eff_timeout,
                               progress=progress)
    except RuntimeError as exc:
        timed_out = (getattr(exc, "kind", "") == "cloud-timeout"
                     or "timed out" in str(exc).lower())
        if fast or not timed_out:
            raise
        if progress:
            progress("Full-resolution segmentation is taking too long on the "
                     "cloud service; retrying in fast (3 mm) mode...", -1.0)
        return _segment_remote(volume, work_dir, name, url, api_key,
                               fast=True, force=force,
                               timeout=_adaptive_timeout(volume, True),
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
                last_err = _cloud_unreachable(f"{label} failed (network): {exc}")
            else:
                if resp.status_code in (200, 201):
                    return resp
                if resp.status_code < 500:
                    raise _cloud_rejected(
                        resp.status_code,
                        f"{label} failed ({resp.status_code}): {resp.text[:300]}")
                last_err = _cloud_unreachable(
                    f"{label} failed ({resp.status_code}): {resp.text[:200]}")
            if i < attempts - 1:
                wait = min(2 ** i, 8) + random.random()
                _report(f"{label}: transient cloud error, retrying in {wait:.0f}s...")
                time.sleep(wait)
        raise last_err or _cloud_unreachable(f"{label} failed")

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
            if st.status_code < 500:
                raise _cloud_rejected(
                    st.status_code,
                    f"status check failed ({st.status_code}): {st.text[:300]}")
            raise _cloud_unreachable(
                f"status check failed ({st.status_code}): {st.text[:300]}")
        info = st.json()
        if info["status"] == "done":
            result_url = info["result_url"]
            break
        if info["status"] == "error":
            raise UserFacingError(
                title="Cloud segmentation failed",
                message="Segmentation failed on the cloud service.",
                remedy="Try again; if this keeps happening, send us the log "
                       "file so we can look at the failed job.",
                detail=f"remote job {job_id} error: {info.get('error')}",
                kind="cloud-failed")
    if result_url is None:
        raise _cloud_unreachable(
            f"remote job {job_id} did not finish within {timeout:.0f}s",
            timed_out=True)

    # 5. download the mask straight from GCS
    _report("Downloading segmentation mask...")
    dl = requests.get(result_url, timeout=600, stream=True)
    if dl.status_code != 200:
        raise _cloud_unreachable(
            f"mask download failed ({dl.status_code}): {dl.text[:300]}")
    with open(seg_path, "wb") as out:
        for chunk in dl.iter_content(chunk_size=1 << 20):
            if chunk:
                out.write(chunk)

    try:
        os.remove(in_path)
    except OSError:
        pass
    return load_segmentation(seg_path)
