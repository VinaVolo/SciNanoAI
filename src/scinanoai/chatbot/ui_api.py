"""HTTP client layer for the Gradio frontend.

Keeps all network access and payload shaping out of the view code in ``ui.py``.
The chatbot API contract (``/chat``, ``/clear_history``, ``/health``) is owned by
:mod:`scinanoai.chatbot.api`; this module only talks to it.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from ..utils.logging import setup_logging

_LOG = setup_logging("scinanoai.ui")

# Generous default: ``/chat`` is synchronous and may run RAG + image analysis.
_CHAT_TIMEOUT = 120.0
_QUICK_TIMEOUT = 10.0


class ChatApiError(RuntimeError):
    """Raised when the chatbot API is unreachable or returns an error.

    Carries a human-readable, Russian message safe to surface in a toast.
    """


def _resolve_path(file_obj: Any) -> str | None:
    """Extract a filesystem path from the many shapes Gradio hands us.

    ``gr.MultimodalTextbox`` yields plain string paths; older ``gr.File`` widgets
    yield dicts or objects exposing ``name`` / ``path``.
    """
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "path", None) or getattr(file_obj, "name", None)


def encode_images(image_files: list[Any] | None) -> list[dict[str, str]]:
    """Base64-encode uploaded image files into the API's ``ImagePayload`` shape.

    Files that cannot be read are logged and skipped rather than aborting the
    whole request.
    """
    encoded: list[dict[str, str]] = []
    for idx, file_obj in enumerate(image_files or [], start=1):
        path = _resolve_path(file_obj)
        if not path:
            continue
        try:
            with open(path, "rb") as fh:
                encoded.append(
                    {
                        "data": base64.b64encode(fh.read()).decode("utf-8"),
                        "name": os.path.basename(path) or f"image_{idx}.png",
                    }
                )
        except OSError as exc:
            _LOG.error("Failed to read image %s: %s", path, exc)
    return encoded


class ChatApiClient:
    """Thin wrapper over the chatbot HTTP API used by the Gradio UI."""

    def __init__(self, base_url: str | None = None, *, timeout: float = _CHAT_TIMEOUT) -> None:
        resolved = base_url or os.getenv("CHAT_API_URL") or "http://localhost:8001"
        self._base_url = resolved.rstrip("/")
        self._timeout = timeout

    def send(
        self,
        message: str,
        image_files: list[Any] | None,
        session_id: str,
    ) -> str:
        """POST a chat turn and return the assistant reply.

        Raises :class:`ChatApiError` on transport or HTTP failure.
        """
        payload = {
            "message": message,
            "images": encode_images(image_files),
            "session_id": session_id,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}/chat", json=payload)
        except httpx.HTTPError as exc:
            _LOG.error("Chat request transport error: %s", exc)
            raise ChatApiError("Не удалось связаться с сервисом чат-бота.") from exc

        if response.status_code != 200:
            _LOG.error("Chat API error %s: %s", response.status_code, response.text)
            raise ChatApiError(
                f"Сервис вернул ошибку ({response.status_code}). Попробуйте ещё раз."
            )
        return str(response.json()["reply"])

    def clear(self, session_id: str) -> None:
        """Reset the server-side conversation history for ``session_id``."""
        try:
            with httpx.Client(timeout=_QUICK_TIMEOUT) as client:
                response = client.post(
                    f"{self._base_url}/clear_history",
                    params={"session_id": session_id},
                )
        except httpx.HTTPError as exc:
            _LOG.error("Clear history transport error: %s", exc)
            raise ChatApiError("Не удалось очистить историю на сервере.") from exc

        if response.status_code != 200:
            _LOG.error("Clear history API error %s: %s", response.status_code, response.text)
            raise ChatApiError(
                f"Не удалось очистить историю ({response.status_code})."
            )

    def health(self) -> bool:
        """Return ``True`` when the chatbot API answers ``/health`` with 200."""
        try:
            with httpx.Client(timeout=_QUICK_TIMEOUT) as client:
                response = client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
