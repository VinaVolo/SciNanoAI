"""Build / extend a FAISS index from parsed PDF page JSONs in data/parsed_pages/.

This formalises the pipeline that previously lived only in
``notebooks/parse_papers.ipynb`` and ``notebooks/create_vector_db.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ...utils.logging import setup_logging
from ...utils.paths import get_project_root
from ..repository import IndexManifest

_LOG = setup_logging("scinanoai.ingest.pdf")


def _iter_documents(parsed_pages_dir: Path):
    for path in sorted(parsed_pages_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _LOG.warning("Skipping malformed %s: %s", path.name, exc)
            continue

        # The parsed JSON is expected to be a list of page entries, each with
        # at least a ``text`` field. Accept either shape: list-of-dicts or
        # dict-with-"pages".
        pages = data if isinstance(data, list) else data.get("pages") or []
        for page_idx, page in enumerate(pages):
            text = (page.get("text") or "").strip() if isinstance(page, dict) else ""
            if not text:
                continue
            yield text, {
                "filename": path.name,
                "page": page_idx,
            }


def build_index(
    *,
    parsed_pages_dir: Path,
    db_path: Path,
    model_name: str,
    device: str = "cpu",
) -> int:
    if not parsed_pages_dir.exists():
        raise FileNotFoundError(f"Parsed-pages directory not found: {parsed_pages_dir}")

    texts: list[str] = []
    metadatas: list[dict] = []
    for text, metadata in _iter_documents(parsed_pages_dir):
        texts.append(text)
        metadatas.append(metadata)
    if not texts:
        raise RuntimeError(f"No documents found under {parsed_pages_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    db_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(db_path))

    manifest = IndexManifest(embedding_model=model_name, dimension=store.index.d)
    (db_path / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

    _LOG.info("Built index with %d documents under %s", len(texts), db_path)
    return len(texts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from parsed PDFs")
    parser.add_argument("--parsed-pages", default=Path("data/parsed_pages"), type=Path)
    parser.add_argument("--db-path", default=Path("db/intfloat_multilingual-e5-large"), type=Path)
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    root = get_project_root()
    parsed = args.parsed_pages if args.parsed_pages.is_absolute() else root / args.parsed_pages
    db_path = args.db_path if args.db_path.is_absolute() else root / args.db_path
    build_index(parsed_pages_dir=parsed, db_path=db_path, model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()
