from __future__ import annotations

from typing import Sequence

from langchain_community.chat_models.gigachat import GigaChat

from .base import LLMClient, Message


class GigaChatClient(LLMClient):
    """
    Adapter around the GigaChat LangChain integration.
    """

    def __init__(
        self,
        credentials: str,
        *,
        model_name: str = "GigaChat-lite",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        verify_ssl_certs: bool = False,
    ):
        super().__init__(model_name)
        self._client = GigaChat(
            credentials=credentials,
            verify_ssl_certs=verify_ssl_certs,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_name,
        )

    def generate(self, messages: Sequence[Message], **kwargs) -> str:
        prompt = kwargs.get("prompt_override") or self._format_prompt(messages)
        response = self._client.invoke(prompt)
        content = getattr(response, "content", None)
        return content if isinstance(content, str) else str(response)

    @staticmethod
    def _format_prompt(messages: Sequence[Message]) -> str:
        return "\n".join(f"{message['role'].capitalize()}: {message['content']}" for message in messages)
