"""FastAPI app for the chatbot service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..utils.logging import setup_logging
from .factory import build_chat_service
from .schemas import ChatRequest, ChatResponse
from .service import ChatService

_LOG = setup_logging("scinanoai.chatbot")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.chat_service = build_chat_service()
    _LOG.info("Chatbot service initialised.")
    yield


app = FastAPI(title="SciNanoAI chatbot", version="0.2.0", lifespan=_lifespan)


def _service(app: FastAPI) -> ChatService:
    return app.state.chat_service  # type: ignore[no-any-return]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        reply, history = _service(app).handle(
            request.message,
            images=request.images,
            session_id=request.session_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("Chat request failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal error") from exc
    return ChatResponse(reply=reply, conversation_history=history)


@app.post("/clear_history")
async def clear_history(session_id: str | None = None) -> dict[str, str]:
    _service(app).clear_history(session_id)
    return {"message": "History cleared"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
