"""Token-aware conversation history with optional summarisation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable

import tiktoken

from ..llm.base import LLMMessage

_LOG = logging.getLogger(__name__)

# tiktoken does not ship encoders for every model name; fall back gracefully.
_DEFAULT_ENCODING = "cl100k_base"


def _resolve_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


class ConversationHistory:
    """Append-only conversation log with thread-safe access and token budget."""

    def __init__(
        self,
        *,
        token_budget: int,
        encoder_model: str = "gpt-4o-mini",
        summarize_after_messages: int = 4,
    ) -> None:
        self._messages: list[LLMMessage] = []
        self._lock = threading.Lock()
        self._budget = token_budget
        self._encoding = _resolve_encoding(encoder_model)
        self._summarize_after_messages = summarize_after_messages

    # -- read / write -------------------------------------------------------
    def snapshot(self) -> list[LLMMessage]:
        with self._lock:
            return list(self._messages)

    def append(self, message: LLMMessage) -> None:
        with self._lock:
            self._messages.append(message)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    # -- token accounting ---------------------------------------------------
    def count_tokens(self, messages: Iterable[LLMMessage] | None = None) -> int:
        target = messages if messages is not None else self.snapshot()
        return sum(len(self._encoding.encode(m.content)) for m in target)

    def limit_to_budget(
        self, messages: list[LLMMessage], budget: int | None = None
    ) -> list[LLMMessage]:
        """Drop oldest messages until the running total fits ``budget`` tokens."""
        budget = budget if budget is not None else self._budget
        total = 0
        limited: list[LLMMessage] = []
        for message in reversed(messages):
            message_tokens = len(self._encoding.encode(message.content))
            if total + message_tokens > budget:
                break
            limited.insert(0, message)
            total += message_tokens
        return limited

    # -- summarisation ------------------------------------------------------
    def summarize_if_needed(self, summarizer: Callable[[list[LLMMessage]], str]) -> None:
        with self._lock:
            if self.count_tokens(self._messages) <= self._budget:
                return
            keep_n = self._summarize_after_messages
            if len(self._messages) <= keep_n:
                return
            older = self._messages[:-keep_n]
            recent = self._messages[-keep_n:]
        summary = summarizer(older)
        _LOG.info("History summarised: kept %d recent messages.", len(recent))
        with self._lock:
            self._messages = [
                LLMMessage(
                    role="system", content=f"Summary of the previous conversation: {summary}"
                ),
                *recent,
            ]


class HistoryStore:
    """Per-session ConversationHistory registry (multi-user safe)."""

    def __init__(
        self,
        *,
        token_budget: int,
        encoder_model: str = "gpt-4o-mini",
        summarize_after_messages: int = 4,
    ) -> None:
        self._registry: dict[str, ConversationHistory] = {}
        self._lock = threading.Lock()
        self._token_budget = token_budget
        self._encoder_model = encoder_model
        self._summarize_after_messages = summarize_after_messages

    def get(self, session_id: str) -> ConversationHistory:
        with self._lock:
            history = self._registry.get(session_id)
            if history is None:
                history = ConversationHistory(
                    token_budget=self._token_budget,
                    encoder_model=self._encoder_model,
                    summarize_after_messages=self._summarize_after_messages,
                )
                self._registry[session_id] = history
            return history

    def clear(self, session_id: str) -> None:
        with self._lock:
            history = self._registry.get(session_id)
            if history is not None:
                history.clear()
