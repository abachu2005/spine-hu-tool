"""FastAPI segmentation service.

Endpoints
  GET  /health           -> liveness probe (used by Cloud Run)
  POST /upload-url        -> mint a V4 signed PUT URL so the client can upload a
                             large CT volume straight to GCS (bypasses Cloud
                             Run's 32 MiB request limit).
  POST /runs/upload-url   -> mint a signed PUT URL for one file of an archived
                             run (runs/<YYYY-MM>/<run_id>/<filename>); clients
                             archive inputs/outputs/logs of every run for QA
                             and remote troubleshooting.
  POST /segment-async     -> start segmentation of a GCS-resident volume in a
                             background thread; returns {job_id} immediately.
  GET  /segment-status/.. -> poll a job; when done, returns a signed GET URL to
                             download the (small) mask straight from GCS.
  POST /segment-gcs       -> synchronous variant (kept for tests / short runs);
                             returns the mask in the response body.
  POST /segment           -> direct multipart upload for small volumes / local
                             testing (subject to the 32 MiB limit).

Why async + polling: full-resolution segmentation can take many minutes on CPU.
Holding one HTTP connection open that long is fragile -- idle connections get
dropped by Cloud Run / intermediate proxies, hanging the client even though the
work succeeds. So the long step runs in a background thread that writes the mask
to GCS, and the client only ever makes short requests (start + poll + a direct
GCS download). The heavy work is a TotalSegmentator subprocess, so it doesn't
hold the GIL and status polls stay responsive.

Auth (optional): if SPINE_HU_API_KEY is set, requests to the segment/upload-url
endpoints must send `Authorization: Bearer <key>`. The signed URLs themselves
are authorized by GCS, so the big upload/download need no app key.

Only the CT volume crosses the wire; all measurement/QC/review stays on the
client. This is the only component that needs real RAM/GPU.
"""
from __future__ import annotations
import os
import re
import uuid
import shutil
import datetime
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from ..io.dicom_loader import load_volume_from_nifti
from ..segmentation.totalseg_runner import run_segmentation
from .. import __version__

app = FastAPI(title="Spine HU Segmentation Service", version=__version__)

API_KEY = os.environ.get("SPINE_HU_API_KEY")
BUCKET = os.environ.get("SPINE_HU_BUCKET")
MAX_UPLOAD_MB = float(os.environ.get("SPINE_HU_MAX_UPLOAD_MB", "512"))
SIGN_URL_MINUTES = int(os.environ.get("SPINE_HU_SIGN_MINUTES", "30"))

# In-memory job registry. Cloud Run is pinned to a single instance, so every
# poll lands on the same process that owns these jobs.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _check_auth(authorization: str | None):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ---- GCS helpers (lazy imports so non-GCS use / tests don't need the lib) ----

def _bucket():
    if not BUCKET:
        raise HTTPException(status_code=500, detail="SPINE_HU_BUCKET not configured")
    from google.cloud import storage
    return storage.Client().bucket(BUCKET)


def _signed_url(object_name: str, method: str, content_type: str | None = None) -> str:
    """V4 signed URL, signed via IAM signBlob (no private key needed on Cloud
    Run -- uses the runtime service account + the IAM Credentials API)."""
    from google.auth import default
    from google.auth.transport.requests import Request
    creds, _ = default()
    creds.refresh(Request())
    blob = _bucket().blob(object_name)
    kwargs = dict(
        version="v4",
        expiration=datetime.timedelta(minutes=SIGN_URL_MINUTES),
        method=method,
        service_account_email=getattr(creds, "service_account_email", None),
        access_token=creds.token,
    )
    if content_type is not None:
        kwargs["content_type"] = content_type
    return blob.generate_signed_url(**kwargs)


def _signed_put_url(object_name: str, content_type: str) -> str:
    return _signed_url(object_name, "PUT", content_type)


def _segment_worker(job_id: str, object_name: str, fast: bool):
    """Background thread: download CT from GCS -> segment -> upload mask to GCS.

    Updates the shared job record so the client's status polls can see progress.
    The heavy step is a subprocess, so this thread stays GIL-friendly.
    """
    tmp = tempfile.mkdtemp(prefix="seg_")
    in_path = os.path.join(tmp, "input.nii.gz")
    try:
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "running"
        bucket = _bucket()
        bucket.blob(object_name).download_to_filename(in_path)
        seg_path = _run(tmp, in_path, fast)
        result_object = f"results/{job_id}.nii.gz"
        bucket.blob(result_object).upload_from_filename(seg_path)
        try:
            bucket.blob(object_name).delete()   # uploaded CT no longer needed
        except Exception:
            pass
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="done", result_object=result_object)
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error", error=str(e)[:500])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- models ----

class UploadUrlResponse(BaseModel):
    object_name: str
    url: str
    content_type: str


class RunUploadUrlRequest(BaseModel):
    run_id: str
    filename: str


class SegmentRequest(BaseModel):
    object_name: str
    fast: bool = True


class StartResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    status: str                       # pending | running | done | error
    error: str | None = None
    result_url: str | None = None     # signed GET URL when status == done


# ---- endpoints ----

@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "gcs": bool(BUCKET)}


@app.post("/upload-url", response_model=UploadUrlResponse)
def upload_url(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    object_name = f"uploads/{uuid.uuid4().hex}.nii.gz"
    content_type = "application/octet-stream"
    return UploadUrlResponse(object_name=object_name,
                             url=_signed_put_url(object_name, content_type),
                             content_type=content_type)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_component(value: str, *, max_len: int = 120) -> str:
    """Collapse a client-supplied name to a safe single path component.

    Strips path separators / traversal so a malicious or buggy client can only
    ever write inside its own ``runs/<month>/<run_id>/`` prefix.
    """
    value = os.path.basename(value.replace("\\", "/")).strip()
    value = _SAFE_NAME.sub("_", value).strip("._")
    return value[:max_len]


@app.post("/runs/upload-url", response_model=UploadUrlResponse)
def runs_upload_url(body: RunUploadUrlRequest,
                    authorization: str | None = Header(default=None)):
    """Mint a signed PUT URL for one file of an archived run.

    Every client run (local or cloud segmentation) archives its inputs,
    outputs, and logs under ``runs/<YYYY-MM>/<run_id>/`` for QA and remote
    troubleshooting. The manifest is just another uploaded file
    (``manifest.json``) -- no DB, no server-side run state.
    """
    _check_auth(authorization)
    run_id = _sanitize_component(body.run_id)
    filename = _sanitize_component(body.filename)
    if not run_id or not filename:
        raise HTTPException(status_code=400, detail="invalid run_id or filename")
    month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    object_name = f"runs/{month}/{run_id}/{filename}"
    content_type = "application/octet-stream"
    return UploadUrlResponse(object_name=object_name,
                             url=_signed_put_url(object_name, content_type),
                             content_type=content_type)


@app.post("/segment-async", response_model=StartResponse)
def segment_async(body: SegmentRequest,
                  authorization: str | None = Header(default=None)):
    """Kick off segmentation in the background and return a job id immediately."""
    _check_auth(authorization)
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "pending"}
    t = threading.Thread(target=_segment_worker,
                         args=(job_id, body.object_name, body.fast), daemon=True)
    t.start()
    return StartResponse(job_id=job_id)


@app.get("/segment-status/{job_id}", response_model=StatusResponse)
def segment_status(job_id: str,
                   authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        job = dict(job)
    if job["status"] == "done":
        return StatusResponse(status="done",
                              result_url=_signed_url(job["result_object"], "GET"))
    return StatusResponse(status=job["status"], error=job.get("error"))


@app.post("/segment-gcs")
def segment_gcs(body: SegmentRequest,
                authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    tmp = tempfile.mkdtemp(prefix="seg_")
    in_path = os.path.join(tmp, "input.nii.gz")
    blob = _bucket().blob(body.object_name)
    try:
        blob.download_to_filename(in_path)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=404,
                            detail=f"could not fetch upload: {e}")
    seg_path = _run(tmp, in_path, body.fast)
    blob.delete()   # uploaded CT no longer needed
    return FileResponse(
        seg_path, media_type="application/gzip", filename="seg.nii.gz",
        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))


@app.post("/segment")
async def segment(file: UploadFile = File(...),
                  fast: bool = Form(True),
                  authorization: str | None = Header(default=None)):
    """Direct multipart path (small volumes / local testing; 32 MiB cap)."""
    _check_auth(authorization)
    tmp = tempfile.mkdtemp(prefix="seg_")
    in_path = os.path.join(tmp, "input.nii.gz")
    size = 0
    with open(in_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(tmp, ignore_errors=True)
                raise HTTPException(status_code=413, detail="upload too large")
            f.write(chunk)
    seg_path = _run(tmp, in_path, fast)
    return FileResponse(
        seg_path, media_type="application/gzip", filename="seg.nii.gz",
        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))


def _run(tmp: str, in_path: str, fast: bool) -> str:
    """Load -> segment -> return mask path (cleans up tmp on failure)."""
    try:
        volume = load_volume_from_nifti(in_path)
        run_segmentation(volume, tmp, "case", fast=fast, force=True)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"segmentation failed: {e}")
    seg_path = os.path.join(tmp, "case_seg.nii.gz")
    if not os.path.exists(seg_path):
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=500, detail="segmentation produced no output")
    return seg_path


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))   # Cloud Run sets $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
