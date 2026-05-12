"""FastAPI app for the vector retrieval microservice."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from ..utils.logging import setup_logging
from ..utils.paths import get_project_root
from .repository import VectorRepository
from .schemas import Document, HealthResponse, QueryRequest, QueryResponse, StatsResponse
from .settings import VectorSettings

_LOG = setup_logging("scinanoai.vector_service")


def _resolve_index_path(settings: VectorSettings):
    path = settings.vector_db_path
    if path.is_absolute():
        return path
    return get_project_root() / path


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = VectorSettings()
    repository = VectorRepository(
        index_path=_resolve_index_path(settings),
        embedding_model=settings.vector_embedding_model,
        device=settings.vector_embedding_device,
    )
    repository.load()
    app.state.repository = repository
    app.state.settings = settings
    yield


app = FastAPI(title="SciNanoAI vector service", version="0.2.0", lifespan=_lifespan)


def require_api_key(
    settings: VectorSettings = Depends(lambda: app.state.settings),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = settings.vector_service_api_key
    if expected is None:
        return  # auth disabled
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    repo: VectorRepository = app.state.repository
    return HealthResponse(status="ok", index_loaded=repo.loaded)


@app.get("/v1/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
async def stats() -> StatsResponse:
    repo: VectorRepository = app.state.repository
    return StatsResponse(
        num_documents=repo.num_documents,
        embedding_model=repo.embedding_model,
        index_path=str(repo.index_path),
    )


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
)
async def query(request: QueryRequest) -> QueryResponse:
    repo: VectorRepository = app.state.repository
    settings: VectorSettings = app.state.settings

    k = min(request.k, settings.max_k)
    fetch_k = min(request.fetch_k, settings.max_fetch_k)

    try:
        docs = repo.query(request.query, k=k, fetch_k=fetch_k, lambda_mult=request.lambda_mult)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("Vector query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Vector query failed") from exc

    return QueryResponse(
        documents=[Document(content=d.page_content, metadata=dict(d.metadata)) for d in docs]
    )


# Legacy alias kept for backward compatibility with chatbot_app/main.py shims.
@app.post("/query", response_model=QueryResponse)
async def legacy_query(request: QueryRequest) -> QueryResponse:
    return await query(request)
