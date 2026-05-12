# ------------------------- builder ------------------------------------------
# NOTE: no `# syntax=docker/dockerfile:...` directive on purpose. The BuildKit
# frontend bundled with Docker 23+ already supports `--mount=type=cache`, and
# pulling the dockerfile frontend image from docker.io is fragile on
# restricted networks.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first (better layer caching). --no-install-project avoids
# requiring the source tree until the next step.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra chatbot

COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra chatbot

# ------------------------- runtime ------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    SCINANO_LOG_LEVEL=INFO

RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/configs /app/configs

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8001/health', timeout=3.0).status_code == 200 else 1)"

CMD ["uvicorn", "scinanoai.chatbot.api:app", "--host", "0.0.0.0", "--port", "8001"]
