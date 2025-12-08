from __future__ import annotations

from typing import Any, Dict, List

import requests


class VectorDatabaseClient:
    """
    Small HTTP client around the vector service API.
    """

    def __init__(self, base_url: str, *, timeout_seconds: int = 30):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def query(self, query: str, k: int, lambda_mult: float, fetch_k: int) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "k": k,
            "lambda_mult": lambda_mult,
            "fetch_k": fetch_k,
        }
        response = requests.post(
            f"{self._base_url}/query",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("documents", [])
