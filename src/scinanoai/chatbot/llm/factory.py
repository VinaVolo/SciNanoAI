"""Factory building the LLM client for the OpenAI-compatible gateway."""

from __future__ import annotations

from ..settings import ChatbotSettings
from .base import LLMClient
from .openai_client import OpenAIChatClient


def get_client(settings: ChatbotSettings) -> LLMClient:
    """Return the LLM client pointed at the configured gateway.

    The whole bot runs through one OpenAI-compatible gateway (LiteLLM / OpenRouter /
    vLLM); the gateway resolves ``llm_model`` to its real upstream, so there is a
    single code path here and no model-name routing.
    """
    if not settings.llm_base_url:
        raise ValueError("LLM_BASE_URL is required")
    return OpenAIChatClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key or "",
        base_url=settings.llm_base_url,
    )
