"""LLM client protocol and shared message type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "user" | "assistant" | "system"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion contract shared by every provider."""

    model: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Return the assistant reply text for the given message list."""
        ...
