from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import tiktoken

Message = Dict[str, str]


class TokenCounter:
    """
    Utility that provides token counting/limiting for chat messages.
    """

    def __init__(self, model_name: str):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count(self, messages: Sequence[Message]) -> int:
        return sum(len(self.encoding.encode(message["content"])) for message in messages)

    def limit(self, messages: Sequence[Message], max_tokens: int) -> List[Message]:
        total_tokens = 0
        limited: List[Message] = []
        for message in reversed(messages):
            message_tokens = len(self.encoding.encode(message["content"]))
            if total_tokens + message_tokens > max_tokens:
                break
            limited.insert(0, message)
            total_tokens += message_tokens
        return limited


class ConversationHistory:
    """
    Stores and maintains the rolling conversation between user and assistant.
    """

    def __init__(self, token_counter: TokenCounter, max_tokens: int):
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._messages: List[Message] = []

    @property
    def messages(self) -> List[Message]:
        return list(self._messages)

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_system_message(self, content: str) -> None:
        self._messages.append({"role": "system", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self._messages.clear()

    def summarize_if_needed(
        self,
        summarizer: Callable[[List[Message]], str],
        summary_window: int,
    ) -> None:
        total_tokens = self._token_counter.count(self._messages)
        if total_tokens <= self._max_tokens or len(self._messages) <= summary_window:
            return

        messages_to_summarize = self._messages[:-summary_window]
        recent_messages = self._messages[-summary_window:]
        summary = summarizer(messages_to_summarize)
        if summary:
            summary_message = {
                "role": "system",
                "content": f"Резюме предыдущего разговора: {summary}",
            }
            self._messages = [summary_message] + recent_messages

    def build_prompt_messages(self, prompt: str) -> List[Message]:
        combined = self._messages + [{"role": "user", "content": prompt}]
        return self._token_counter.limit(combined, self._max_tokens)
