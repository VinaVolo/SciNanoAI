"""Factory selecting an LLM client by model name."""

from __future__ import annotations

from ..settings import ChatbotSettings
from .base import LLMClient
from .gigachat_client import GigaChatClient
from .ollama_client import OllamaChatClient
from .openai_client import OpenAIChatClient
from .yandex_client import YandexChatClient


def get_client(settings: ChatbotSettings) -> LLMClient:
    """Return a configured LLM client matching ``settings.llm_model``."""
    model = settings.llm_model

    if model == "YandexGPT4":
        return YandexChatClient(
            api_key=settings.yandex_api_key or "",
            model_uri=settings.yandex_api_base or "",
        )
    if model == "GigaChat-Pro":
        return GigaChatClient(credentials=settings.sber_api_key or "")
    if model == "gpt-oss:latest" or settings.local_api_base:
        return OllamaChatClient(
            model="gpt-oss:latest" if model == "gpt-oss:latest" else model,
            api_base=settings.local_api_base or "",
            api_key=settings.local_api_key or "",
        )
    return OpenAIChatClient(
        model=model,
        api_key=settings.openai_api_key or "",
        base_url=settings.openai_api_base,
    )
