import os
from dataclasses import dataclass


def _get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(f"Environment variable '{name}' is required but was not provided.")
    return value or ""


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class ChatbotSettings:
    """
    Centralized immutable configuration for the chatbot application.
    """

    llm_model: str
    analysis_model: str
    openai_api_key: str
    openai_api_base: str
    yandex_api_key: str
    yandex_api_base: str
    sber_api_key: str
    vector_service_url: str
    chat_api_url: str
    tokenizer_model: str
    max_model_tokens: int
    max_reply_tokens: int
    summary_history_window: int
    vector_top_k: int
    vector_lambda_mult: float
    vector_fetch_k: int
    decomposer_threshold: float
    vector_timeout_seconds: int
    ollama_api_base: str
    ollama_api_key: str
    ollama_jwt_token: str

    @classmethod
    def from_env(cls) -> "ChatbotSettings":
        openai_api_key = _get_env("OPENAI_API_KEY", "")
        default_llm = "openai/gpt-4o-mini" if openai_api_key else "gpt-oss:latest"

        return cls(
            llm_model=_get_env("LLM_MODEL", default_llm),
            analysis_model=_get_env("ANALYSIS_MODEL_NAME", default_llm),
            openai_api_key=openai_api_key,
            openai_api_base=_get_env("OPENAI_API_BASE", "https://api.openai.com/v1"),
            yandex_api_key=_get_env("YANDEX_API_KEY", ""),
            yandex_api_base=_get_env("YANDEX_API_BASE", ""),
            sber_api_key=_get_env("SBER_API_KEY", ""),
            vector_service_url=_get_env("VECTOR_SERVICE_URL", "http://localhost:8000"),
            chat_api_url=_get_env("CHAT_API_URL", "http://localhost:8001"),
            tokenizer_model=_get_env("TOKEN_MODEL_NAME", "gpt-4o-mini"),
            max_model_tokens=_get_int_env("MAX_MODEL_TOKENS", 4096),
            max_reply_tokens=_get_int_env("MAX_REPLY_TOKENS", 1000),
            summary_history_window=_get_int_env("SUMMARY_HISTORY_WINDOW", 4),
            vector_top_k=_get_int_env("VECTOR_TOP_K", 10),
            vector_lambda_mult=_get_float_env("VECTOR_LAMBDA_MULT", 0.45),
            vector_fetch_k=_get_int_env("VECTOR_FETCH_K", 50),
            decomposer_threshold=_get_float_env("DECOMPOSER_THRESHOLD", 0.2),
            vector_timeout_seconds=_get_int_env("VECTOR_TIMEOUT_SECONDS", 30),
            ollama_api_base=_get_env("OLLAMA_API_BASE", "https://chat.itmo.shockofwave.su/api/chat/completions"),
            ollama_api_key=_get_env("OLLAMA_API_KEY", ""),
            ollama_jwt_token=_get_env("OLLAMA_JWT_TOKEN", ""),
        )

    @property
    def max_allowed_tokens(self) -> int:
        return max(self.max_model_tokens - self.max_reply_tokens - 100, 0)
