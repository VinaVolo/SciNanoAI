"""GigaChat / Sber chat client (langchain-community wrapper)."""

from __future__ import annotations

from typing import Sequence

from langchain_community.chat_models.gigachat import GigaChat

from .base import LLMMessage


class GigaChatClient:
    model = "GigaChat-Pro"

    def __init__(self, *, credentials: str, model_name: str = "GigaChat-Pro") -> None:
        if not credentials:
            raise ValueError("SBER_API_KEY is required for GigaChat client")
        self._credentials = credentials
        self._model_name = model_name

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        llm = GigaChat(
            credentials=self._credentials,
            verify_ssl_certs=False,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self._model_name,
        )
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for m in messages:
            if m.role == "user":
                lc_messages.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                lc_messages.append(AIMessage(content=m.content))
            else:
                lc_messages.append(SystemMessage(content=m.content))
        return llm.invoke(lc_messages).content
