"""Best-effort cloud archival of every analysis run (local AND cloud seg).

Every run's inputs, outputs, and logs are uploaded to the hosted service's GCS
bucket under ``runs/<YYYY-MM>/<run_id>/`` so any user-reported problem can be
diagnosed remotely -- what data they ran, what came out, and the full logs --
without asking for screenshots, and so there is a record of all data/outputs.

Archival is strictly best-effort and asynchronous:
  - it runs in a daemon thread, so it never slows down or blocks an analysis;
  - every failure is swallowed (logged only), so a machine that is offline or a
    server hiccup can NEVER break or fail a run;
  - ``SPINE_HU_NO_ARCHIVE=1`` disables it entirely (developer/test kill
    switch; also set by the test-suite to keep tests hermetic).

The server side is one endpoint (``POST /runs/upload-url``) minting signed PUT
URLs; the manifest is just another uploaded file -- no database.
"""
from __future__ import annotations
import datetime as _dt
import getpass
import hashlib
import json
import os
import platform
import socket
import tempfile
import threading
import traceback
import uuid
from typing import Optional


def _log():
    from ..app.applog import get_logger
    return get_logger("archive")


def archiving_disabled() -> bool:
    return os.environ.get("SPINE_HU_NO_ARCHIVE", "") not in ("", "0")


def new_run_id() -> str:
    """Sortable, collision-safe id: UTC timestamp + short random suffix."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _series_uid_hash(series_uid: Optional[str]) -> Optional[str]:
    """Stable pseudonymous id for the series (never the raw UID)."""
    if not series_uid:
        return None
    return hashlib.sha256(str(series_uid).encode()).hexdigest()[:16]


class RunArchiver:
    """Uploads one run's files to ``runs/<YYYY-MM>/<run_id>/`` in the bucket.

    One instance per analysis run; ``run_id`` ties the analysis-time archive
    and the later export/finalize archive together.
    """

    def __init__(self, seg_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 run_id: Optional[str] = None):
        self.run_id = run_id or new_run_id()
        self._seg_url = seg_url
        self._api_key = api_key

    # -- public API (all fire-and-forget) ----------------------------------

    def submit(self, manifest: dict, files: Optional[dict] = None) -> None:
        """Upload ``manifest.json`` + the given ``{remote_name: local_path}``
        files in a background daemon thread. Never raises, never blocks."""
        if archiving_disabled():
            return
        t = threading.Thread(
            target=self._upload_all, args=(dict(manifest), dict(files or {})),
            name=f"run-archiver-{self.run_id}", daemon=True)
        t.start()

    def base_manifest(self, *, backend: str, status: str, **extra) -> dict:
        """Manifest skeleton shared by success and failure archives."""
        from .. import __version__
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        m = {
            "schema_version": 2,
            "inclusion_policy": "qc-fail-default-excluded-explicit-include",
            "run_id": self.run_id,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "app_version": __version__,
            "hostname": socket.gethostname(),
            "os_user": user,
            "platform": platform.platform(),
            "backend": backend,               # "local" | "cloud"
            "status": status,                 # "ok" | "failed"
        }
        m.update(extra)
        return m

    # -- internals ----------------------------------------------------------

    def _endpoint(self) -> Optional[str]:
        from ..segmentation.backends import resolve_seg_url
        return resolve_seg_url(self._seg_url)

    def _upload_all(self, manifest: dict, files: dict) -> None:
        try:
            url = self._endpoint()
            if not url:
                _log().info("archive skipped (no service URL): run %s", self.run_id)
                return
            import requests
            from ..segmentation.backends import resolve_api_key
            key = resolve_api_key(self._api_key)
            headers = {"Authorization": f"Bearer {key}"} if key else {}

            def _put(name: str, data=None, path: str = None) -> None:
                r = requests.post(f"{url}/runs/upload-url", headers=headers,
                                  json={"run_id": self.run_id, "filename": name},
                                  timeout=60)
                r.raise_for_status()
                up = r.json()
                put_headers = {"Content-Type": up["content_type"]}
                if path is not None:
                    with open(path, "rb") as fh:
                        pr = requests.put(up["url"], data=fh,
                                          headers=put_headers, timeout=600)
                else:
                    pr = requests.put(up["url"], data=data,
                                      headers=put_headers, timeout=600)
                pr.raise_for_status()

            uploaded = []
            for name, path in files.items():
                try:
                    if path and os.path.exists(path):
                        _put(name, path=path)
                        uploaded.append(name)
                except Exception as e:      # per-file: skip and continue
                    _log().warning("archive: failed to upload %s for run %s: %s",
                                   name, self.run_id, e)
            manifest["archived_files"] = uploaded
            _put("manifest.json",
                 data=json.dumps(manifest, indent=2, default=str).encode())
            _log().info("archived run %s (%d files)", self.run_id, len(uploaded))
        except Exception as e:              # best-effort: never propagate
            _log().warning("archive failed for run %s: %s", self.run_id, e)


# ---- high-level hooks used by analyze_dataset / export ------------------------

def archive_analysis_success(archiver: RunArchiver, state, *, backend: str,
                             mode: str, fast, series, work_dir: str,
                             name: str, timings: Optional[dict] = None) -> None:
    """Archive a completed analysis: manifest + input volume + mask + results.

    Called from ``analyze_dataset`` on success (GUI, CLI, and batch runs all
    pass through it). Everything heavy (writing the int16 input NIfTI) happens
    inside the archiver's daemon thread, not here.
    """
    if archiving_disabled():
        return
    try:
        manifest = archiver.base_manifest(
            backend=backend, status="ok", mode=mode,
            fast=bool(fast),
            series_uid_hash=_series_uid_hash(getattr(series, "series_uid", None)),
            n_slices=getattr(series, "n_files", None),
            seg_status=state.case.get("seg_status"),
            levels={lvl: r.summary() for lvl, r in state.case["results"].items()},
            timings=timings or {},
        )

        files: dict = {}
        seg_path = os.path.join(work_dir, f"{name}_seg.nii.gz")
        if os.path.exists(seg_path):
            files["seg.nii.gz"] = seg_path
        seg_log = os.path.join(work_dir, f"{name}_seg.log")
        if os.path.exists(seg_log):
            files["totalseg.log"] = seg_log

        # The input volume: cloud runs delete their upload file after use and
        # local runs may hit the cache, so write a fresh int16 NIfTI (lossless
        # for CT HU) into a temp dir -- inside the daemon thread, because
        # serializing a full CT is too slow to put on the caller's path.
        volume = state.volume

        def _submit_with_input():
            tmp = tempfile.mkdtemp(prefix="spinehu-archive-")
            try:
                import numpy as np
                from ..io.nifti_io import save_volume_nifti
                in_path = os.path.join(tmp, "input.nii.gz")
                save_volume_nifti(volume, in_path, dtype=np.int16)
                files["input.nii.gz"] = in_path
            except Exception as e:
                _log().warning("archive: could not write input volume: %s", e)
            archiver._upload_all(manifest, files)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

        threading.Thread(target=_submit_with_input,
                         name=f"run-archiver-{archiver.run_id}",
                         daemon=True).start()
    except Exception as e:                   # hooks must never break analysis
        _log().warning("archive (success) hook failed: %s", e)


def archive_analysis_failure(archiver: RunArchiver, exc: BaseException, *,
                             backend: str, mode: str, fast,
                             series=None, work_dir: Optional[str] = None,
                             name: Optional[str] = None,
                             folder: Optional[str] = None) -> None:
    """Archive a failed analysis: manifest with the error + app/seg logs.

    This is the remote-troubleshooting payload: exactly what the user saw
    (message/remedy for UserFacingError), the full traceback, and the logs.
    """
    if archiving_disabled():
        return
    try:
        from ..errors import UserFacingError
        err_info = {"type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(exc))}
        if isinstance(exc, UserFacingError):
            err_info.update(exc.to_manifest())
        else:
            err_info["message"] = str(exc)
        manifest = archiver.base_manifest(
            backend=backend, status="failed", mode=mode, fast=bool(fast),
            series_uid_hash=_series_uid_hash(getattr(series, "series_uid", None)),
            source_folder_name=os.path.basename(folder or "") or None,
            error=err_info,
        )
        files: dict = {}
        try:
            from ..app.applog import log_path
            if os.path.exists(log_path()):
                files["spine_hu.log"] = log_path()
        except Exception:
            pass
        if work_dir and name:
            seg_log = os.path.join(work_dir, f"{name}_seg.log")
            if os.path.exists(seg_log):
                files["totalseg.log"] = seg_log
        archiver.submit(manifest, files)
    except Exception as e:
        _log().warning("archive (failure) hook failed: %s", e)


def archive_export(state_case: dict, export_dir: str,
                   written: Optional[dict] = None) -> None:
    """Archive the reviewed/final export under the run's id.

    Called after :func:`export.writers.export_case`; uploads the physician's
    final numbers (measurements, reproducibility record, audit trail) so the
    reviewed results are on record next to the raw run.
    """
    if archiving_disabled():
        return
    try:
        run_id = state_case.get("archive_run_id") or new_run_id()
        archiver = RunArchiver(run_id=run_id)
        manifest = archiver.base_manifest(backend="n/a", status="ok",
                                          stage="export")
        files = {}
        for fname in ("measurements.csv", "measurements.json",
                      "reproducibility.json", "comparison.json",
                      "audit_trail.json"):
            p = os.path.join(export_dir, fname)
            if os.path.exists(p):
                files[fname] = p
        if written:
            for key, p in written.items():
                if isinstance(p, str) and os.path.isfile(p):
                    files.setdefault(os.path.basename(p), p)
        archiver.submit(manifest, files)
    except Exception as e:
        _log().warning("archive (export) hook failed: %s", e)
