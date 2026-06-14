"""HTTP client for the vector_service retrieval microservice."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedDocument:
    content: str
    metadata: dict[str, Any]


class VectorServiceClient:
    """Thin wrapper around the FastAPI /query endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4.0),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def query(
        self,
        text: str,
        *,
        k: int = 10,
        lambda_mult: float = 0.45,
        fetch_k: int = 50,
    ) -> list[RetrievedDocument]:
        if not text or not text.strip():
            return []

        url = f"{self._base_url}/v1/query"
        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        payload = {"query": text, "k": k, "lambda_mult": lambda_mult, "fetch_k": fetch_k}

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        documents = data.get("documents", [])
        _LOG.debug("Vector service returned %d documents", len(documents))
        return [
            RetrievedDocument(
                content=str(doc.get("content", "")),
                metadata=dict(doc.get("metadata") or {}),
            )
            for doc in documents
        ]
