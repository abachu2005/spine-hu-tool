# `server/` — the segmentation service

A small FastAPI service that runs the one heavy step (TotalSegmentator) off the
user's machine. Deployable to Google Cloud Run (scale-to-zero). See
[`deploy/DEPLOY.md`](../../deploy/DEPLOY.md) for deployment.

| File | What it does |
|------|--------------|
| `app.py` | FastAPI app: health, signed-URL minting, async segmentation + polling, and direct/synchronous fallbacks. |

## Why a service at all

TotalSegmentator needs real RAM/GPU and OOMs ~8 GB laptops. Offloading it keeps
the desktop app lean and installable by anyone, and lets one capable host serve
a whole lab. **Only the CT volume crosses the wire; all measurement, QC, and
review stay on the client.** This is the only component that needs serious
hardware.

## Why signed URLs + async polling (not a plain upload)

Two hard constraints shaped the API:

1. **Cloud Run caps request bodies at 32 MiB**, but CT volumes are much larger.
   So the client asks for a **V4 signed PUT URL** (`/upload-url`) and uploads the
   volume **straight to GCS**, bypassing the service entirely. The mask comes
   back the same way via a signed GET URL.
2. **Full-resolution CPU segmentation takes minutes**, and a single long-held
   HTTP connection gets dropped by Cloud Run / proxies, hanging the client even
   when the work succeeds. So `/segment-async` starts the job in a background
   thread and returns a `job_id`; the client makes only **short** requests
   (`/segment-status/{job_id}` polls). The heavy work is a subprocess, so it
   doesn't hold the GIL and polls stay responsive.

`/segment-gcs` (synchronous) and `/segment` (direct multipart, 32 MiB cap) are
kept for tests and short local runs.

## Auth model

If `SPINE_HU_API_KEY` is set, the app endpoints require `Authorization: Bearer
<key>` (a shared pilot key). The big upload/download need no app key because the
**signed URLs are authorized by GCS itself**. Signing uses IAM `signBlob` via the
runtime service account, so no private key file lives on the server.
