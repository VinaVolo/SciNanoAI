import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from models import Document, QueryRequest, QueryResponse
from vector_db import VectorDatabase


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "db" / "intfloat_multilingual-e5-large"


@dataclass(frozen=True)
class VectorServiceSettings:
    """
    Holds configuration for the vector service.
    """

    db_path: str = os.getenv("VECTOR_DB_PATH") or str(DEFAULT_DB_PATH)
    model_name: str = os.getenv("VECTOR_MODEL_NAME", "intfloat/multilingual-e5-large")

    def __post_init__(self) -> None:
        if not self.db_path:
            raise ValueError("VECTOR_DB_PATH must be provided.")
        if not self.model_name:
            raise ValueError("VECTOR_MODEL_NAME must be provided.")


class VectorService:
    """
    Facade that exposes vector DB functionality to the API layer.
    """

    def __init__(self, settings: VectorServiceSettings, vector_db: Optional[VectorDatabase] = None):
        self.settings = settings
        self.vector_db = vector_db or VectorDatabase(
            db_path=settings.db_path,
            model_name=settings.model_name,
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        documents = self.vector_db.query(
            request.query,
            request.k,
            request.lambda_mult,
            request.fetch_k,
        )
        response_documents = [
            Document(content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        return QueryResponse(documents=response_documents)

    def get_document_count(self) -> int:
        return self.vector_db.get_num_documents()


class VectorAPI:
    """
    Binds FastAPI routes to service methods.
    """

    def __init__(self, service: VectorService):
        self.service = service
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.post("/query", response_model=QueryResponse)
        def query_vector_db(request: QueryRequest):
            return self.service.query(request)

        @self.app.get("/num_documents")
        def get_num_documents():
            return {"num_documents": self.service.get_document_count()}


settings = VectorServiceSettings()
service = VectorService(settings=settings)
vector_api = VectorAPI(service=service)
app = vector_api.app
