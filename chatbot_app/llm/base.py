from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

Message = Dict[str, str]


class LLMClient(ABC):
    """
    Abstract chat model interface.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, messages: Sequence[Message], **kwargs) -> str:
        """
        Produces a reply for the provided conversation.
        """
        raise NotImplementedError
