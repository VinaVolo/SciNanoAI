"""Unit tests for the chatbot Router."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scinanoai.chatbot.core.router import ImageMode, Router, looks_like_image_question


@dataclass
class _StubDecomposer:
    verdict: bool

    def should_use_database(self, _: str) -> bool:
        return self.verdict


@dataclass
class _StubFormality:
    formal: bool

    def is_formal(self, _: str) -> bool:
        return self.formal


@dataclass
class _StubImageDecision:
    response: str

    def decide(self, _q: str, _notes: str) -> str:
        return self.response


@pytest.mark.unit
def test_no_images_uses_decomposer_only() -> None:
    router = Router(decomposer=_StubDecomposer(verdict=True))
    decision = router.route("question", has_images=False)
    assert decision.use_database is True
    assert decision.image_mode is ImageMode.NO_IMAGES


@pytest.mark.unit
def test_image_keyword_forces_image_analysis() -> None:
    router = Router(
        decomposer=_StubDecomposer(verdict=True),
        formality_agent=_StubFormality(formal=True),
        image_decision_agent=_StubImageDecision(response="literature"),
    )
    decision = router.route("Что на этом изображении?", has_images=True, image_notes="x")
    assert decision.image_mode is ImageMode.IMAGE_ANALYSIS
    assert decision.use_database is False


@pytest.mark.unit
def test_informal_question_with_images() -> None:
    router = Router(
        decomposer=_StubDecomposer(verdict=False),
        formality_agent=_StubFormality(formal=False),
        image_decision_agent=_StubImageDecision(response="literature"),
    )
    decision = router.route("hi there", has_images=True, image_notes="x")
    assert decision.image_mode is ImageMode.INFORMAL


@pytest.mark.unit
def test_formal_question_literature_branch() -> None:
    router = Router(
        decomposer=_StubDecomposer(verdict=False),
        formality_agent=_StubFormality(formal=True),
        image_decision_agent=_StubImageDecision(response="literature"),
    )
    decision = router.route("Compare ZnO nanostructures", has_images=True, image_notes="x")
    assert decision.use_database is True
    assert decision.image_mode is ImageMode.LITERATURE


@pytest.mark.unit
def test_missing_agents_raises_when_images_present() -> None:
    router = Router(decomposer=_StubDecomposer(verdict=True))
    with pytest.raises(RuntimeError):
        router.route("question", has_images=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Look at this image", True),
        ("Что на картинке?", True),
        ("Some photo here", True),
        ("Just a plain question", False),
        ("", False),
    ],
)
def test_looks_like_image_question(text: str, expected: bool) -> None:
    assert looks_like_image_question(text) is expected
