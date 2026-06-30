# Segmentation service for Cloud Run (CPU). TotalSegmentator + FastAPI.
# Python 3.12: required by the pinned numpy/scipy (>=3.12) in requirements-server.txt.
FROM python:3.12-slim

# System libs TotalSegmentator/SimpleITK/nibabel need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 git curl && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # Cache the downloaded TotalSegmentator weights inside the image layer at
    # build time (see below) so cold starts don't re-download.
    TOTALSEG_HOME_DIR=/opt/totalseg

WORKDIR /app

COPY deploy/requirements-server.txt .
# Install CPU-only torch AND the matching torchvision from the same index so
# they're an ABI-matched pair (a mismatch breaks TotalSegmentator at import,
# e.g. torchvision::nms register_fake errors). Keep the CPU index as an extra
# index for the rest so nothing pulls a CUDA/mismatched build from PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-server.txt

# App code
COPY spine_hu_tool ./spine_hu_tool
COPY pyproject.toml README.md ./
RUN pip install -e . --no-deps

# Pre-download model weights into the image so cold starts don't pay the
# download cost: 297 = fast (3 mm); 291-295 = full-resolution (1.5 mm) 'total'
# ensemble. Best-effort: if it fails, weights download on the first request.
RUN python -c "from totalsegmentator.libs import download_pretrained_weights; [download_pretrained_weights(t) for t in (297, 291, 292, 293, 294, 295)]" \
    || echo "WARN: weight pre-download skipped; will download on first request"

EXPOSE 8080
CMD ["python", "-m", "spine_hu_tool.server.app"]
