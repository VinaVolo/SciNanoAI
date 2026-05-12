"""YandexGPT chat client (langchain-community wrapper)."""

from __future__ import annotations

from typing import Sequence

from langchain_community.chat_models.yandex import ChatYandexGPT

from .base import LLMMessage


class YandexChatClient:
    model = "YandexGPT4"

    def __init__(self, *, api_key: str, model_uri: str, model_name: str = "yandexgpt-32k") -> None:
        if not api_key or not model_uri:
            raise ValueError("YANDEX_API_KEY and YANDEX_API_BASE are required")
        self._api_key = api_key
        self._model_uri = model_uri
        self._model_name = model_name

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        llm = ChatYandexGPT(
            api_key=self._api_key,
            model_uri=self._model_uri,
            model_name=self._model_name,
            temperature=temperature,
            max_tokens=max_tokens,
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
