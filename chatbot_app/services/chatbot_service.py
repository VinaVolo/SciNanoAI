from __future__ import annotations

import logging
from typing import List

import os
import pandas as pd

from ..config import ChatbotSettings
from ..core import prompts
from ..core.history import ConversationHistory
from ..llm.base import LLMClient
from src.utils.paths import get_project_path
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
        self._reference_mapping = self._load_reference_mapping()

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
        reply = self._replace_reference_links(reply)
        self._history.add_assistant_message(reply)

        if self._should_retry(reply, question, attempt):
            reformulated_question = self._rephraser.rephrase(question).strip()
            if self._is_valid_reformulation(question, reformulated_question):
                return self._respond(reformulated_question, attempt + 1)

        return reply

    def _should_retry(self, reply: str, question: str, attempt: int) -> bool:
        if attempt >= self._max_rephrase_attempts:
            return False
        return not self._evaluator.is_answer_complete(reply, question)

    def _is_valid_reformulation(self, original: str, reformulated: str) -> bool:
        if not reformulated:
            return False
        if reformulated.strip().lower() == original.strip().lower():
            return False
        placeholder_phrases = (
            "пожалуйста", "уточните", "напишите", "к сожалению", "вопрос не указан"
        )
        lowered = reformulated.lower()
        if any(lowered.startswith(prefix) for prefix in placeholder_phrases):
            return False
        if "?" not in reformulated and len(reformulated.split()) < 6:
            return False
        return True

    def _load_reference_mapping(self) -> pd.DataFrame | None:
        data_path = os.path.join(get_project_path(), "data", "updated_references_links.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None

    def extract_bracket_content(self, text: str) -> List[str]:
        import re

        return re.findall(r"\[([^\[\]]+)\]", text)

    def _replace_reference_links(self, reply: str) -> str:
        if self._reference_mapping is None:
            return reply
        df = self._reference_mapping
        found_links = self.extract_bracket_content(reply)
        new_answer = reply
        for link in found_links:
            for filename in df["filename"].values:
                if link in filename:
                    new_link = df[df["filename"] == filename].iloc[0].link_name
                    new_answer = new_answer.replace(link, new_link)
        return new_answer

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
