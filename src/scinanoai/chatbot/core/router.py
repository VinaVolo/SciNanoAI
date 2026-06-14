"""Routing logic: text/image, RAG vs direct, formality detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ImageMode(str, Enum):
    NO_IMAGES = "no_images"
    IMAGE_ANALYSIS = "image_analysis"
    LITERATURE = "literature"
    INFORMAL = "informal"


@dataclass(frozen=True)
class RouteDecision:
    use_database: bool
    image_mode: ImageMode


_IMAGE_KEYWORDS: tuple[str, ...] = (
    "изображ",
    "картин",
    "фото",
    "image",
    "picture",
    "photo",
)


def looks_like_image_question(question: str) -> bool:
    q = (question or "").lower()
    return any(keyword in q for keyword in _IMAGE_KEYWORDS)


class Router:
    """Decides how a turn should be processed.

    The contract is intentionally simple: ``route`` is a pure function over
    side-effect-free callables. Concrete strategies (decomposer / formality
    agent / image-decision agent) are supplied via dependency injection, which
    keeps the routing decision deterministic and unit-testable.
    """

    def __init__(
        self,
        *,
        decomposer,  # has should_use_database(question) -> bool
        formality_agent=None,  # has is_formal(question) -> bool
        image_decision_agent=None,  # has decide(question, notes) -> str
    ) -> None:
        self._decomposer = decomposer
        self._formality = formality_agent
        self._image_decision = image_decision_agent

    def route(self, question: str, *, has_images: bool, image_notes: str = "") -> RouteDecision:
        use_database = self._decomposer.should_use_database(question)

        if not has_images:
            return RouteDecision(use_database=use_database, image_mode=ImageMode.NO_IMAGES)

        if self._formality is None or self._image_decision is None:
            raise RuntimeError("Image uploads require an LLM client (LLM_BASE_URL / LLM_API_KEY).")

        if looks_like_image_question(question):
            return RouteDecision(use_database=False, image_mode=ImageMode.IMAGE_ANALYSIS)

        if not self._formality.is_formal(question):
            return RouteDecision(use_database=use_database, image_mode=ImageMode.INFORMAL)

        decision = self._image_decision.decide(question, image_notes)
        if decision == "image_analysis":
            return RouteDecision(use_database=False, image_mode=ImageMode.IMAGE_ANALYSIS)
        return RouteDecision(use_database=True, image_mode=ImageMode.LITERATURE)
