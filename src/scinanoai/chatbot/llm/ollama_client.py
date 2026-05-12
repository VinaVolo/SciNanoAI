"""Local Ollama / gpt-oss HTTP chat client."""

from __future__ import annotations

import logging
from typing import Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMMessage

_LOG = logging.getLogger(__name__)


class OllamaChatClient:
    def __init__(self, *, model: str, api_base: str, api_key: str, timeout: float = 120.0) -> None:
        if not api_base or not api_key:
            raise ValueError("LOCAL_API_BASE and LOCAL_API_KEY are required for Ollama client")
        self.model = model
        self._api_base = api_base
        self._api_key = api_key
        self._timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4.0),
    )
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._api_base, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            _LOG.error("Unexpected response shape from Ollama: %s", data)
            raise RuntimeError("Malformed response from local LLM endpoint") from exc
