from __future__ import annotations
from typing import List, Sequence

import requests

from .base import LLMClient, Message


class OllamaChatClient(LLMClient):
    """
    Minimal client for interacting with Ollama-compatible /api/chat endpoints.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        *,
        timeout_seconds: int = 60,
    ):
        super().__init__(model_name)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def _serialize_messages(self, messages: Sequence[Message]) -> List[Message]:
        serialized: List[Message] = []
        for message in messages:
            role = message.get("role") or "user"
            content = message.get("content") or ""
            serialized.append({"role": role, "content": content})
        return serialized

    def generate(self, messages: Sequence[Message], **kwargs) -> str:
        msg_list = self._serialize_messages(messages)
        options = {}
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            options["num_predict"] = kwargs["max_tokens"]

        payload = {
            "model": self.model_name,
            "messages": msg_list,
            "stream": False,  # We return the result at once
        }
        if options:
            payload["options"] = options

        response = requests.post(
            self._base_url,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict):
            if isinstance(data.get("message"), dict):
                return str(data["message"].get("content", "")).strip()
            if "response" in data:
                return str(data["response"]).strip()
        return str(data).strip()
