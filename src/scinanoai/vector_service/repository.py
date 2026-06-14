"""FAISS-backed vector repository with index manifest checks."""

from __future__ import annotations

import json
import logging
import time
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
        _LOG.info("VectorRepository.load() starting. index_path=%s", self._index_path)
        t0 = time.monotonic()

        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        _LOG.info("Step 1/4: checking index path %s ...", self._index_path)
        if not self._index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self._index_path}")

        _LOG.info("Step 2/4: reading manifest.json ...")
        manifest = IndexManifest.load(self._index_path / "manifest.json")
        if manifest:
            _LOG.info(
                "manifest found: embedding_model=%s dimension=%s",
                manifest.embedding_model,
                manifest.dimension,
            )
            if manifest.embedding_model != self._embedding_model:
                raise RuntimeError(
                    "Embedding-model mismatch: index built with "
                    f"{manifest.embedding_model!r} but service configured for "
                    f"{self._embedding_model!r}. Re-run the ingest pipeline."
                )
        else:
            _LOG.warning(
                "No manifest.json next to the index — cannot verify embedding model. "
                "Continuing on user trust."
            )

        _LOG.info(
            "Step 3/4: initialising HuggingFaceEmbeddings model=%r device=%s "
            "(first run downloads ~2 GB from huggingface.co; subsequent runs hit ~/.cache/huggingface)",
            self._embedding_model,
            self._device,
        )
        t_emb = time.monotonic()
        embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._device},
        )
        _LOG.info("Embeddings ready in %.1fs", time.monotonic() - t_emb)

        _LOG.info("Step 4/4: loading FAISS index from %s ...", self._index_path)
        t_idx = time.monotonic()
        self._store = FAISS.load_local(
            str(self._index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _LOG.info(
            "Vector index loaded: %s (%d vectors) in %.1fs. Total startup time: %.1fs",
            self._index_path,
            self.num_documents,
            time.monotonic() - t_idx,
            time.monotonic() - t0,
        )

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
