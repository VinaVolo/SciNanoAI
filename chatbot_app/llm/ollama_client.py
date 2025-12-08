from __future__ import annotations

from typing import Sequence

import requests

from .base import LLMClient, Message


class OllamaChatClient(LLMClient):
    """
    Minimal OpenAI-compatible client for locally hosted models (e.g., gpt-oss via Ollama).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        jwt_token: str | None = None,
        *,
        default_temperature: float = 0.2,
        default_max_tokens: int = 4096,
        timeout_seconds: int = 60,
    ):
        super().__init__(model_name)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""
        self._jwt_token = jwt_token or ""
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._timeout = timeout_seconds

    def generate(self, messages: Sequence[Message], **kwargs) -> str:
        temperature = kwargs.get("temperature", self._default_temperature)
        max_tokens = kwargs.get("max_tokens", self._default_max_tokens)
        payload = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._jwt_token:
            headers["Authorization"] = f"Bearer {self._jwt_token}"
        # if self._api_key:
        #     headers["X-API-Key"] = self._api_key

        print(f"{self._base_url}")
        
        response = requests.post(
            f"{self._base_url}",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        return content.strip()
