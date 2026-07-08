# Deploying the segmentation service to Google Cloud Run

The heavy ML step (TotalSegmentator) runs here; the desktop app stays light and
just calls this service. Cloud Run **scales to zero**, so you pay only while a
scan is actually being segmented.

## Prerequisites
- A GCP project with billing enabled (new accounts get $300 free credit).
- `gcloud` CLI installed and authenticated: `gcloud auth login`.
- Enable services:
  ```bash
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
  ```

## 1. Set variables
```bash
export PROJECT=$(gcloud config get-value project)
export REGION=us-central1
export SERVICE=spine-hu-seg
export API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(24))")   # save this
echo "API key: $API_KEY"
```

## 2. Build the image (from the repo root)
```bash
gcloud builds submit --tag gcr.io/$PROJECT/$SERVICE -f deploy/Dockerfile .
```

## 3. Deploy to Cloud Run
TotalSegmentator on CPU needs RAM; give it 16 GB / 4 vCPU. The model is baked
into the image, and segmentation can take a few minutes, so allow a long timeout
and scale-to-zero.
```bash
gcloud run deploy $SERVICE \
  --image gcr.io/$PROJECT/$SERVICE \
  --region $REGION \
  --memory 16Gi --cpu 4 \
  --no-cpu-throttling \
  --timeout 900 \
  --concurrency 1 \
  --min-instances 0 --max-instances 3 \
  --set-env-vars SPINE_HU_API_KEY=$API_KEY \
  --no-allow-unauthenticated
```
- `--concurrency 1`: one big job per instance (segmentation is memory-heavy).
- `--no-cpu-throttling` (**important**): segmentation runs in a background thread
  after `/segment-async` returns, and the client then makes only short status
  polls. With the Cloud Run default ("CPU allocated only during request
  processing") that background worker is throttled to ~0 CPU between polls, so a
  full-resolution job crawls -- a ~115-slice scan took ~18 min and a ~250-slice
  scan blew past the client deadline entirely. Keeping the CPU always allocated
  lets the worker run at full speed (seconds-to-minutes). The desktop client also
  now falls back to fast (3 mm) mode if a full-res job still times out, but that
  is a safety net -- fix the throttling here for good full-res performance.
- `--min-instances 0`: scale to zero = no idle cost (first request after idle
  pays a cold start; the model is pre-baked so it's just container start).
- `--no-allow-unauthenticated` + `SPINE_HU_API_KEY`: keep it private.

> Speed tip: for the spine use-case only the vertebra/sacrum labels are needed,
> so passing `--roi_subset` (vertebrae + sacrum) to TotalSegmentator would cut
> full-res runtime further. Not enabled yet (kept as the full multilabel run) to
> avoid changing cached-mask contents mid-pilot.

> Optional GPU (faster, ~seconds/scan): add `--gpu 1 --gpu-type nvidia-l4`
> and build with a CUDA torch wheel instead of the CPU one.

## 4. Get the URL and test
```bash
export URL=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')
# health (needs an identity token because the service is private):
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" $URL/health
```

## 5. Point the desktop app at it
On the laptop, set the endpoint + key (or paste the URL into the app's landing
screen field):
```bash
export SPINE_HU_SEG_URL=$URL
export SPINE_HU_API_KEY=$API_KEY
python -m spine_hu_tool.app.viewer
```
Now **Analyze** uploads the CT to Cloud Run, segments there, and downloads the
mask — the laptop only does the light deterministic measurement + review.

## HIPAA note
For real patient data, sign a **BAA with Google Cloud** (covers Cloud Run, GCS,
Artifact Registry), require auth (done above), and de-identify before upload
(DICOM tags **and** burned-in pixel PHI). De-identification is currently out of
scope in this repo and must be added before clinical/product use.
