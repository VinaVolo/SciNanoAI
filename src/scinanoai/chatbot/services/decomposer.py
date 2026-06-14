"""Zero-shot topic classifier deciding whether to consult the vector DB."""

from __future__ import annotations

import logging
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)


_DEFAULT_TOPICS: tuple[str, ...] = (
    "photoresistors",
    "types of nanostructures",
    "machine learning in nanostructuring",
    "nanostructuring methods",
    "materials for two-photon polymerization",
    "the effect of nanostructuring on cells",
    "the effect of nanoparticles on cells",
    "nanotubes and nanopores in cellular technologies",
    "the effect of nanostructuring on cell adhesion",
    "the effect of nanostructuring on cell differentiation",
    "the effect of nanostructuring on cellular forces",
    "the effect of nanostructuring on cell migration",
    "the effect of nanostructuring on cell proliferation",
    "Another topic",
)

# A multilingual NLI checkpoint is more appropriate as a zero-shot classifier
# than the prior reranker (BAAI/bge-reranker-large) used by the legacy code.
_DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@dataclass
class DecomposerAgent:
    threshold: float = 0.2
    model_name: str = _DEFAULT_MODEL
    topics: tuple[str, ...] = _DEFAULT_TOPICS

    def __post_init__(self) -> None:
        # Lazy import keeps the chatbot importable even when transformers
        # is not installed (e.g., in unit tests).
        from transformers import pipeline

        _LOG.info("Loading zero-shot classifier model=%s", self.model_name)
        self._classifier = pipeline("zero-shot-classification", model=self.model_name)

    def should_use_database(self, question: str) -> bool:
        if not question or not question.strip():
            return False
        result = self._classifier(question, list(self.topics))
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        _LOG.debug("Decomposer top_label=%s score=%.3f", top_label, top_score)
        if top_label == "Another topic" or top_score < self.threshold:
            return False
        return True
