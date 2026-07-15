# DVXR Screen — deployable HTTP API image.
#
# Research-grade clinical-risk screening (NOT a diagnosis). Offline / CPU / deterministic:
# the container needs no network at runtime. LaBraM weights are baked in at build time so the
# depression path works air-gapped; band-power tasks need no weights.
#
#   docker build -t dvxr-screen .
#   docker run --rm -p 8000:8000 dvxr-screen
#   curl localhost:8000/health
#
# Screeners are read from /app/outputs/product/screeners/<task>/. Mount your own to override:
#   docker run --rm -p 8000:8000 -v "$PWD/outputs:/app/outputs" dvxr-screen
FROM python:3.11-slim

# Keep BLAS/OMP from thrashing on shared / constrained hosts (see project thread-cap note).
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Deps first for layer caching: copy metadata, install the eeg+io+api extras, then the source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[eeg,io,api]"

# Fitted screeners so the API serves without re-training at boot.
COPY outputs/product/screeners ./outputs/product/screeners

# Bake the real LaBraM safetensors into the HF cache at build time so the depression path runs
# offline. Build-time network only; skip with --build-arg BAKE_LABRAM=0 (band-power tasks like
# wesad_stress still work fully offline without weights).
ARG BAKE_LABRAM=1
RUN if [ "$BAKE_LABRAM" = "1" ]; then \
      DVXR_LABRAM_ALLOW_DOWNLOAD=1 python -c \
      "from dvxr.encoders.labram_real import LaBraMEncoder; LaBraMEncoder.from_pretrained()" ; \
    fi

EXPOSE 8000
# Bind 0.0.0.0 so the port is reachable from outside the container.
CMD ["python", "-m", "uvicorn", "dvxr.serve.api:app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
