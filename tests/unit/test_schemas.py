"""Unit tests for FastAPI request schemas (bounds enforcement)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scinanoai.vector_service.schemas import QueryRequest


@pytest.mark.unit
def test_query_request_defaults() -> None:
    req = QueryRequest(query="hello")
    assert req.k == 5
    assert 0 <= req.lambda_mult <= 1


@pytest.mark.unit
def test_query_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="")


@pytest.mark.unit
def test_query_request_clamps_via_validation() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="x", k=9999)
    with pytest.raises(ValidationError):
        QueryRequest(query="x", lambda_mult=2.0)
