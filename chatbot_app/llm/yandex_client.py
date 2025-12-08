from __future__ import annotations

from typing import Sequence

from langchain_community.chat_models.yandex import ChatYandexGPT

from .base import LLMClient, Message


class YandexGPTClient(LLMClient):
    """
    Adapter for the LangChain YandexGPT chat model.
    """

    def __init__(
        self,
        api_key: str,
        model_uri: str,
        *,
        model_name: str = "yandexgpt-32k",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        super().__init__(model_name)
        self._client = ChatYandexGPT(
            api_key=api_key,
            model_uri=model_uri,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, messages: Sequence[Message], **kwargs) -> str:
        prompt = kwargs.get("prompt_override") or self._format_prompt(messages)
        result = self._client.invoke(prompt)
        content = getattr(result, "content", None)
        return content if isinstance(content, str) else str(result)

    @staticmethod
    def _format_prompt(messages: Sequence[Message]) -> str:
        return "\n".join(f"{message['role'].capitalize()}: {message['content']}" for message in messages)
