# syntax=docker/dockerfile:1.7
# ------------------------- builder ------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install --target=/install ".[vector]"

# ------------------------- runtime ------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/usr/local/lib/python3.11/site-packages \
    SCINANO_LOG_LEVEL=INFO

RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY --chown=appuser:appuser src ./src

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=3.0).status_code == 200 else 1)"

CMD ["uvicorn", "scinanoai.vector_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
