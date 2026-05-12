"""FAISS-backed vector repository with index manifest checks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)


@dataclass
class IndexManifest:
    embedding_model: str
    dimension: int | None = None

    def to_json(self) -> str:
        return json.dumps({"embedding_model": self.embedding_model, "dimension": self.dimension})

    @classmethod
    def load(cls, path: Path) -> IndexManifest | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(embedding_model=data["embedding_model"], dimension=data.get("dimension"))


class VectorRepository:
    """Loads the FAISS index lazily and exposes query/stats."""

    def __init__(
        self,
        *,
        index_path: Path,
        embedding_model: str,
        device: str = "cpu",
    ) -> None:
        self._index_path = Path(index_path)
        self._embedding_model = embedding_model
        self._device = device
        self._store = None  # FAISS instance

    # ------------------------------------------------------------ lifecycle
    def load(self) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        if not self._index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self._index_path}")

        manifest = IndexManifest.load(self._index_path / "manifest.json")
        if manifest and manifest.embedding_model != self._embedding_model:
            raise RuntimeError(
                "Embedding-model mismatch: index built with "
                f"{manifest.embedding_model!r} but service configured for "
                f"{self._embedding_model!r}. Re-run the ingest pipeline."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._device},
        )
        self._store = FAISS.load_local(
            str(self._index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _LOG.info("Vector index loaded: %s (%d vectors)", self._index_path, self.num_documents)

    @property
    def loaded(self) -> bool:
        return self._store is not None

    @property
    def num_documents(self) -> int:
        if self._store is None:
            return 0
        return int(self._store.index.ntotal)

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    # --------------------------------------------------------------- query
    def query(
        self,
        text: str,
        *,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> list[Any]:
        if self._store is None:
            raise RuntimeError("Repository not loaded; call .load() first")
        retriever = self._store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "lambda_mult": lambda_mult, "fetch_k": fetch_k},
        )
        return retriever.invoke(text)
