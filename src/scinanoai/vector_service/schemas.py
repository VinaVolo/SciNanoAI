"""Vector service DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=50)
    fetch_k: int = Field(default=50, ge=1, le=200)
    lambda_mult: float = Field(default=0.45, ge=0.0, le=1.0)


class Document(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    documents: list[Document]


class StatsResponse(BaseModel):
    num_documents: int
    embedding_model: str
    index_path: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
