"""Chatbot runtime configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatbotSettings(BaseSettings):
    """All configuration knobs read from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM
    llm_model: str = Field(default="gpt-oss:latest")
    openai_api_key: str | None = None
    openai_api_base: str | None = None
    yandex_api_key: str | None = None
    yandex_api_base: str | None = None
    sber_api_key: str | None = None
    local_api_key: str | None = None
    local_api_base: str | None = None

    # --- Vector retrieval
    vector_service_url: str = "http://localhost:8000"
    vector_service_api_key: str | None = None

    # --- Image analysis
    cellpose_model_path: Path = Path("models/cellpose_v0_1/cellpose_full_stream_filtred")
    image_um_per_px: float = 100.0 / 155.0

    # --- History / tokens
    max_model_tokens: int = 4096
    max_reply_tokens: int = 1000
    summarize_after_messages: int = 4

    # --- Gradio
    gradio_username: str | None = None
    gradio_password: str | None = None
    gradio_port: int = 8517
    gradio_root_path: str = "/scinanoai"

    @property
    def history_token_budget(self) -> int:
        return max(self.max_model_tokens - self.max_reply_tokens - 100, 512)
