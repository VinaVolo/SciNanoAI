from __future__ import annotations

from .base import LLMClient
from .gigachat_client import GigaChatClient
from .ollama_client import OllamaChatClient
from .openai_client import OpenAIChatClient
from .yandex_client import YandexGPTClient
from ..config import ChatbotSettings


class LLMFactory:
    """
    Produces ready-to-use chat model clients based on configuration.
    """

    def __init__(self, settings: ChatbotSettings):
        self._settings = settings

    def create_primary_client(self) -> LLMClient:
        return self._create_client(self._settings.llm_model)

    def create_analysis_client(self) -> LLMClient:
        return self._create_client(self._settings.analysis_model)

    def _create_client(self, model: str) -> LLMClient:
        if self._is_ollama_model(model):
            return self._create_ollama(model)
        if model == "YandexGPT4":
            return self._create_yandex()
        if model == "GigaChat-Pro":
            return self._create_gigachat()
        return self._create_openai(model)

    def _create_openai(self, model_name: str) -> LLMClient:
        if not self._settings.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY must be set to use model "
                f"'{model_name}'. Alternatively, configure LLM_MODEL/ANALYSIS_MODEL_NAME "
                "to point to a supported local provider (e.g., gpt-oss:latest)."
            )
        return OpenAIChatClient(
            model_name=model_name,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_api_base,
            default_max_tokens=self._settings.max_reply_tokens,
        )

    def _create_yandex(self) -> LLMClient:
        if not self._settings.yandex_api_key or not self._settings.yandex_api_base:
            raise EnvironmentError(
                "YANDEX_API_KEY and YANDEX_API_BASE must be provided to use YandexGPT."
            )
        return YandexGPTClient(
            api_key=self._settings.yandex_api_key,
            model_uri=self._settings.yandex_api_base,
        )

    def _create_gigachat(self) -> LLMClient:
        if not self._settings.sber_api_key:
            raise EnvironmentError("SBER_API_KEY must be provided to use GigaChat.")
        return GigaChatClient(credentials=self._settings.sber_api_key)

    def _create_ollama(self, model_name: str) -> LLMClient:
        if not self._settings.ollama_api_base:
            raise EnvironmentError("OLLAMA_API_BASE must be provided to use local models.")
        return OllamaChatClient(
            model_name=model_name,
            base_url=self._settings.ollama_api_base,
            timeout_seconds=self._settings.vector_timeout_seconds,
        )

    @staticmethod
    def _is_ollama_model(model_name: str) -> bool:
        prefixes = ("gpt-oss", "ollama/", "ollama:")
        return model_name.startswith(prefixes)
