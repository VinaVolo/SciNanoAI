from __future__ import annotations

import logging
from typing import List

from ..config import ChatbotSettings
from ..core import prompts
from ..core.history import ConversationHistory
from ..llm.base import LLMClient
from .analysis import AnswerEvaluator, ConversationSummarizer, QuestionRephraser
from .decomposer import DecomposerAgent
from .vector_client import VectorDatabaseClient

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    High-level orchestrator that wires together retrieval, prompting and generation.
    """

    def __init__(
        self,
        settings: ChatbotSettings,
        primary_llm: LLMClient,
        analysis_llm: LLMClient,
        vector_client: VectorDatabaseClient,
        conversation_history: ConversationHistory,
        decomposer_agent: DecomposerAgent,
    ):
        self._settings = settings
        self._primary_llm = primary_llm
        self._vector_client = vector_client
        self._history = conversation_history
        self._decomposer = decomposer_agent

        self._summarizer = ConversationSummarizer(analysis_llm)
        self._evaluator = AnswerEvaluator(analysis_llm)
        self._rephraser = QuestionRephraser(analysis_llm)
        self._max_rephrase_attempts = 2

    @property
    def conversation_history(self) -> List[dict]:
        return self._history.messages

    def clear_history(self) -> None:
        self._history.clear()

    def generate_response(self, question: str) -> str:
        return self._respond(question, attempt=0)

    def _respond(self, question: str, attempt: int) -> str:
        self._history.add_user_message(question)
        self._history.summarize_if_needed(
            self._summarizer.summarize,
            self._settings.summary_history_window,
        )

        prompt = self._build_prompt(question)
        prompt_messages = self._history.build_prompt_messages(prompt)
        reply = self._primary_llm.generate(
            prompt_messages,
            max_tokens=self._settings.max_reply_tokens,
        )
        self._history.add_assistant_message(reply)

        if self._should_retry(reply, question, attempt):
            reformulated_question = self._rephraser.rephrase(question)
            return self._respond(reformulated_question, attempt + 1)

        return reply

    def _should_retry(self, reply: str, question: str, attempt: int) -> bool:
        if attempt >= self._max_rephrase_attempts:
            return False
        return not self._evaluator.is_answer_complete(reply, question)

    def _build_prompt(self, question: str) -> str:
        if self._decomposer.should_use_database(question):
            try:
                documents = self._vector_client.query(
                    question,
                    k=self._settings.vector_top_k,
                    lambda_mult=self._settings.vector_lambda_mult,
                    fetch_k=self._settings.vector_fetch_k,
                )
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning("Vector DB query failed, falling back to direct answer: %s", exc)
                documents = []
            if documents:
                context = prompts.format_context_documents(documents)
                return prompts.build_contextual_prompt(question, context)
        return prompts.build_direct_prompt(question)
