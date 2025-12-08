from __future__ import annotations

from .config import ChatbotSettings
from .core.history import ConversationHistory, TokenCounter
from .llm.factory import LLMFactory
from .services.chatbot_service import ChatbotService
from .services.decomposer import DecomposerAgent
from .services.vector_client import VectorDatabaseClient


def create_chatbot_service(settings: ChatbotSettings | None = None) -> ChatbotService:
    """
    Application-level factory that wires together the chatbot service and its dependencies.
    """

    settings = settings or ChatbotSettings.from_env()
    factory = LLMFactory(settings)
    primary_llm = factory.create_primary_client()
    analysis_llm = factory.create_analysis_client()

    token_counter = TokenCounter(settings.tokenizer_model)
    conversation_history = ConversationHistory(token_counter, settings.max_allowed_tokens)

    vector_client = VectorDatabaseClient(
        settings.vector_service_url,
        timeout_seconds=settings.vector_timeout_seconds,
    )
    decomposer = DecomposerAgent(threshold=settings.decomposer_threshold)

    return ChatbotService(
        settings=settings,
        primary_llm=primary_llm,
        analysis_llm=analysis_llm,
        vector_client=vector_client,
        conversation_history=conversation_history,
        decomposer_agent=decomposer,
    )


__all__ = ["ChatbotService", "create_chatbot_service", "ChatbotSettings"]
