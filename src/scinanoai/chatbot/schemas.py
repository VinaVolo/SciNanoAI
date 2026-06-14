"""Pydantic DTOs for the chatbot HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImagePayload(BaseModel):
    data: str = Field(..., description="Base64-encoded image bytes")
    name: str = Field(..., max_length=255)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    images: list[ImagePayload] | None = None
    session_id: str | None = Field(
        default=None,
        description="Optional session key for multi-user history isolation",
        max_length=64,
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    reply: str
    conversation_history: list[ChatMessage]
