"""High-level orchestrator that ties LLM + RAG + image analysis together."""

from __future__ import annotations

import logging

from .core.history import HistoryStore
from .core.prompts import (
    IMAGE_NOTE_TEMPLATE,
    render_image_prompt,
    render_judge_prompt,
    render_plain_prompt,
    render_rag_prompt,
    render_reformulate_prompt,
    render_summarize_prompt,
)
from .core.router import ImageMode, Router
from .llm.base import LLMClient, LLMMessage
from .schemas import ChatMessage, ImagePayload
from .services.image_analysis import ImageAnalyzer
from .services.references import ReferenceResolver
from .services.vector_client import VectorServiceClient
from .settings import ChatbotSettings

_LOG = logging.getLogger(__name__)

_DEFAULT_SESSION = "default"
_MAX_REFORMULATIONS = 1


class ChatService:
    """End-to-end chatbot orchestrator. Stateless across requests except for
    the session-keyed conversation history."""

    def __init__(
        self,
        *,
        settings: ChatbotSettings,
        llm_client: LLMClient,
        local_client: LLMClient | None,
        router: Router,
        vector_client: VectorServiceClient,
        image_analyzer: ImageAnalyzer,
        reference_resolver: ReferenceResolver,
        history_store: HistoryStore,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._local = local_client
        self._router = router
        self._vector = vector_client
        self._images = image_analyzer
        self._references = reference_resolver
        self._history = history_store

    # ------------------------------------------------------------------ API
    def handle(
        self,
        question: str,
        *,
        images: list[ImagePayload] | None = None,
        session_id: str | None = None,
    ) -> tuple[str, list[ChatMessage]]:
        images = images or []
        session_key = session_id or _DEFAULT_SESSION
        history = self._history.get(session_key)
        history.append(LLMMessage(role="user", content=question))
        history.summarize_if_needed(self._summarize)

        image_notes = ""
        if images:
            image_notes = "\n".join(d.text for d in self._images.describe(images))

        decision = self._router.route(question, has_images=bool(images), image_notes=image_notes)
        _LOG.info(
            "Routing decision: use_database=%s image_mode=%s",
            decision.use_database,
            decision.image_mode.value,
        )

        if decision.image_mode is ImageMode.IMAGE_ANALYSIS:
            reply = self._answer_with_images(question, images)
        else:
            reply = self._answer_with_text(
                question, images=images, route=decision
            )

        history.append(LLMMessage(role="assistant", content=reply))
        snapshot = [ChatMessage(role=m.role, content=m.content) for m in history.snapshot()]
        return reply, snapshot

    def clear_history(self, session_id: str | None = None) -> None:
        self._history.clear(session_id or _DEFAULT_SESSION)

    # ----------------------------------------------------------------- impl
    def _answer_with_text(self, question: str, *, images, route) -> str:
        if route.use_database:
            docs = self._vector.query(question)
            context_parts = [
                f"{d.content} [{d.metadata.get('filename', 'unknown_file')}]"
                for d in docs
            ]
            context = "\n\n".join(context_parts) or "(no documents retrieved)"
            prompt = render_rag_prompt(context=context, question=question)
        else:
            prompt = render_plain_prompt(question)

        if images and route.image_mode is not ImageMode.IMAGE_ANALYSIS:
            prompt += IMAGE_NOTE_TEMPLATE.format(count=len(images))

        reply = self._invoke_with_followup(question, prompt)
        return self._references.enrich(reply)

    def _answer_with_images(self, question: str, images: list[ImagePayload]) -> str:
        metrics_text, _metrics = self._images.analyze(images)
        prompt = render_image_prompt(question=question, metrics=metrics_text)
        client = self._local or self._llm
        return client.complete(
            [LLMMessage(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=2048,
        )

    def _invoke_with_followup(
        self,
        question: str,
        prompt: str,
        attempt: int = 0,
    ) -> str:
        messages = self._build_messages(prompt)
        reply = self._llm.complete(
            messages,
            temperature=0.2,
            max_tokens=self._settings.max_reply_tokens,
        )
        if attempt >= _MAX_REFORMULATIONS or self._local is None:
            return reply
        if self._is_insufficient(reply, question):
            _LOG.info("Reply judged insufficient; reformulating question (attempt=%d).", attempt + 1)
            new_question = self._reformulate(question)
            return self._invoke_with_followup(new_question, render_plain_prompt(new_question), attempt + 1)
        return reply

    def _build_messages(self, prompt: str) -> list[LLMMessage]:
        history_snapshot = self._history.get(_DEFAULT_SESSION).snapshot()
        messages = [*history_snapshot, LLMMessage(role="user", content=prompt)]
        return self._history.get(_DEFAULT_SESSION).limit_to_budget(messages)

    # ---- judge / reformulate / summarize use the local model only -------
    def _is_insufficient(self, reply: str, question: str) -> bool:
        if self._local is None:
            return False
        verdict = self._local.complete(
            [LLMMessage(role="user", content=render_judge_prompt(question=question, answer=reply))],
            temperature=0.0,
            max_tokens=10,
        )
        cleaned = verdict.strip().rstrip(".").lower()
        return cleaned == "no"

    def _reformulate(self, question: str) -> str:
        assert self._local is not None
        return self._local.complete(
            [LLMMessage(role="user", content=render_reformulate_prompt(question))],
            temperature=0.5,
            max_tokens=100,
        )

    def _summarize(self, messages: list[LLMMessage]) -> str:
        client = self._local or self._llm
        conversation = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages
        )
        return client.complete(
            [LLMMessage(role="user", content=render_summarize_prompt(conversation))],
            temperature=0.2,
            max_tokens=2048,
        )
