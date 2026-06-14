"""Unit tests for ReferenceResolver."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scinanoai.chatbot.services import references as ref_module
from scinanoai.chatbot.services.references import ReferenceResolver


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    ref_module._load_reference_table.cache_clear()


@pytest.mark.unit
def test_enrich_replaces_known_citations(tmp_path: Path) -> None:
    csv_path = tmp_path / "refs.csv"
    pd.DataFrame(
        {
            "filename": ["paper-001.pdf", "paper-002.pdf"],
            "link_name": ["https://example.com/p1", "https://example.com/p2"],
        }
    ).to_csv(csv_path, index=False)

    resolver = ReferenceResolver(csv_path=csv_path)
    answer = "Background [paper-001.pdf] then [paper-002.pdf]."
    result = resolver.enrich(answer)
    assert "https://example.com/p1" in result
    assert "https://example.com/p2" in result


@pytest.mark.unit
def test_enrich_passthrough_when_csv_missing(tmp_path: Path) -> None:
    resolver = ReferenceResolver(csv_path=tmp_path / "missing.csv")
    assert resolver.enrich("plain answer") == "plain answer"
