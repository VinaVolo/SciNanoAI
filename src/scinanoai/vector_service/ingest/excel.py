"""Append rows from an Excel table into an existing FAISS index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ...utils.logging import setup_logging
from ...utils.paths import get_project_root

_LOG = setup_logging("scinanoai.ingest.excel")

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "material_id",
    "structure_id",
    "class",
    "mean_radius",
    "mean_area",
    "num_objects_in_image",
    "img_area",
    "%, area",
    "cyto/nuclei (cell spreading)",
)


def _format_row(row: pd.Series) -> str:
    return (
        f"material_id: {row['material_id']}; "
        f"structure_id: {row['structure_id']}; "
        f"class: {row['class']}; "
        f"mean_radius: {row['mean_radius']}; "
        f"mean_area: {row['mean_area']}; "
        f"num_objects_in_image: {row['num_objects_in_image']}; "
        f"img_area: {row['img_area']}; "
        f"%, area: {row['%, area']}; "
        f"cyto/nuclei (cell spreading): {row['cyto/nuclei (cell spreading)']}"
    )


def _build_metadata(idx: int, row: pd.Series, source: str) -> dict:
    return {
        "source": source,
        "row_index": int(idx),
        "material_id": row["material_id"],
        "structure_id": row["structure_id"],
        "class": row["class"],
    }


def ingest_excel(*, excel_path: Path, db_path: Path, model_name: str) -> int:
    df = pd.read_excel(excel_path)

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {excel_path}: {missing}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    store = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)

    texts = [_format_row(row) for _, row in df.iterrows()]
    metadatas = [_build_metadata(idx, row, excel_path.name) for idx, row in df.iterrows()]

    store.add_texts(texts=texts, metadatas=metadatas)
    store.save_local(str(db_path))
    _LOG.info("Appended %d rows from %s into %s", len(texts), excel_path, db_path)
    return len(texts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest Excel rows into FAISS")
    parser.add_argument("--excel", required=True, type=Path)
    parser.add_argument("--db-path", default=Path("db/intfloat_multilingual-e5-large"), type=Path)
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    args = parser.parse_args(argv)

    db_path = args.db_path if args.db_path.is_absolute() else get_project_root() / args.db_path
    ingest_excel(excel_path=args.excel, db_path=db_path, model_name=args.model)


if __name__ == "__main__":
    main()
