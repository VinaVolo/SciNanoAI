# ------------------------- builder ------------------------------------------
# NOTE: no `# syntax=docker/dockerfile:...` directive on purpose. The BuildKit
# frontend bundled with Docker 23+ already supports `--mount=type=cache`, and
# pulling the dockerfile frontend image from docker.io is fragile on
# restricted networks.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Selects the torch wheel index: `cpu` (default, ~200 MB) or `gpu` (CUDA 13,
# ~4 GB). Wired from TORCH_VARIANT in docker-compose.yml / .env.
ARG TORCH_VARIANT=cpu

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra vector --extra "${TORCH_VARIANT}"

COPY src ./src
# pyproject.toml declares `license = { file = "LICENSE" }` and
# `readme = "README.md"`; hatchling validates both at build time.
COPY LICENSE README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra vector --extra "${TORCH_VARIANT}"

# ------------------------- runtime ------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    SCINANO_LOG_LEVEL=INFO

RUN useradd --create-home --shell /bin/bash --uid 10001 appuser \
 && install -d -o appuser -g appuser -m 0755 /home/appuser/.cache/huggingface

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=3.0).status_code == 200 else 1)"

CMD ["uvicorn", "scinanoai.vector_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
