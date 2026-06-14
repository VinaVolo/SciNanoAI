"""Vector service configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vector_db_path: Path = Path("db/intfloat_multilingual-e5-large")
    vector_embedding_model: str = "intfloat/multilingual-e5-large"
    vector_embedding_device: str = "cpu"
    vector_service_api_key: str | None = Field(default=None, description="Optional X-API-Key value")

    # Server-side limits (caller-provided values are clamped to these bounds).
    max_k: int = 50
    max_fetch_k: int = 200
    max_query_length: int = 4000
