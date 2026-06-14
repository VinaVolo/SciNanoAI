"""Lightweight LLM-driven heuristics for formality + image decision."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMClient, LLMMessage


@dataclass
class FormalityAgent:
    """Decides whether the user query is a strict scientific request."""

    client: LLMClient

    def is_formal(self, question: str) -> bool:
        prompt = (
            "Определи, является ли запрос формальным научно-техническим вопросом, "
            "который требует строгого ответа. Ответь одним словом: 'formal' или 'informal'.\n"
            f"Запрос: {question}"
        )
        verdict = self.client.complete(
            [LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=10,
        ).lower()
        return verdict.startswith("formal")


@dataclass
class ImageDecisionAgent:
    """Given a formal question + image notes, decide literature vs image analysis."""

    client: LLMClient

    def decide(self, question: str, image_notes: str) -> str:
        prompt = (
            "Пользователь задал формальный вопрос и загрузил изображения.\n"
            "Нужно определить дальнейший план:\n"
            "- Ответь 'image_analysis', если необходим детальный анализ изображений.\n"
            "- Ответь 'literature', если достаточно поиска литературы/текстовых источников.\n"
            f"Описание изображений:\n{image_notes}\n\n"
            f"Вопрос: {question}"
        )
        verdict = self.client.complete(
            [LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=10,
        ).lower()
        if verdict.startswith("image"):
            return "image_analysis"
        return "literature"
