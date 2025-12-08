from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from .base import LLMClient, Message


class OpenAIChatClient(LLMClient):
    """
    Thin wrapper around the official OpenAI SDK to match the internal interface.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        *,
        default_temperature: float = 0.2,
        default_max_tokens: int = 4096,
    ):
        super().__init__(model_name)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    def generate(self, messages: Sequence[Message], **kwargs) -> str:
        temperature = kwargs.get("temperature", self._default_temperature)
        max_tokens = kwargs.get("max_tokens", self._default_max_tokens)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
