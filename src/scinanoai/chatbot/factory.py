"""Wire-up: build a ChatService instance from environment-driven settings."""

from __future__ import annotations

from pathlib import Path

from ..utils.paths import get_project_root
from .core.history import HistoryStore
from .core.router import Router
from .llm.factory import get_client
from .llm.ollama_client import OllamaChatClient
from .service import ChatService
from .services.decomposer import DecomposerAgent
from .services.formality import FormalityAgent, ImageDecisionAgent
from .services.image_analysis import ImageAnalyzer
from .services.references import ReferenceResolver
from .services.vector_client import VectorServiceClient
from .settings import ChatbotSettings


def _resolve_cellpose_path(settings: ChatbotSettings) -> Path | None:
    path = Path(settings.cellpose_model_path)
    if path.is_absolute():
        return path if path.exists() else None
    root = get_project_root()
    candidate = root / path
    return candidate if candidate.exists() else None


def build_chat_service(settings: ChatbotSettings | None = None) -> ChatService:
    settings = settings or ChatbotSettings()

    primary_llm = get_client(settings)

    local_llm = None
    if settings.local_api_base and settings.local_api_key:
        local_llm = OllamaChatClient(
            model="gpt-oss:latest",
            api_base=settings.local_api_base,
            api_key=settings.local_api_key,
        )

    formality_agent = FormalityAgent(local_llm) if local_llm else None
    image_decision_agent = ImageDecisionAgent(local_llm) if local_llm else None

    decomposer = DecomposerAgent(threshold=0.2)
    router = Router(
        decomposer=decomposer,
        formality_agent=formality_agent,
        image_decision_agent=image_decision_agent,
    )

    vector_client = VectorServiceClient(
        base_url=settings.vector_service_url,
        api_key=settings.vector_service_api_key,
    )

    image_analyzer = ImageAnalyzer(
        cellpose_model_path=_resolve_cellpose_path(settings),
        image_um_per_px=settings.image_um_per_px,
    )

    history_store = HistoryStore(
        token_budget=settings.history_token_budget,
        encoder_model="gpt-4o-mini",
        summarize_after_messages=settings.summarize_after_messages,
    )

    return ChatService(
        settings=settings,
        llm_client=primary_llm,
        local_client=local_llm,
        router=router,
        vector_client=vector_client,
        image_analyzer=image_analyzer,
        reference_resolver=ReferenceResolver(),
        history_store=history_store,
    )
