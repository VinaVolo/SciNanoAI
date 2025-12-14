"""
Append rows from an Excel table into the existing FAISS vector store.

Usage:
    .venv/bin/python vector_service/ingest_excel_to_db.py \
        --excel /Users/vinavolo/Public/Project/SciNanoAI/data/img_result_nn_radius_cyto_2.xlsx \
        --db-path /Users/vinavolo/Public/Project/SciNanoAI/db/intfloat_multilingual-e5-large \
        --model intfloat/multilingual-e5-large
"""

import argparse
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_db(db_path: Path, model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    store = FAISS.load_local(
        str(db_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store


def format_row(row: pd.Series) -> str:
    return (
        f"material_id: {row['material_id']}; "
        f"structure_id: {row['structure_id']}; "
        f"magnification: {row['magnification']}; "
        f"mean_radius: {row['mean_radius']}; "
        f"mean_area: {row['mean_area']}; "
        f"class: {row['class']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to Excel file")
    parser.add_argument(
        "--db-path",
        default="/Users/vinavolo/Public/Project/SciNanoAI/db/intfloat_multilingual-e5-large",
        help="Path to FAISS directory",
    )
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-large",
        help="Embedding model name",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    db_path = Path(args.db_path)

    df = pd.read_excel(excel_path)
    texts = [format_row(row) for _, row in df.iterrows()]
    metadatas = []
    for idx, row in df.iterrows():
        metadatas.append(
            {
                "source": excel_path.name,
                "row_index": int(idx),
                "material_id": row["material_id"],
                "structure_id": row["structure_id"],
                "magnification": row["magnification"],
                "class": row["class"],
            }
        )

    store = load_db(db_path, args.model)
    store.add_texts(texts=texts, metadatas=metadatas)
    store.save_local(str(db_path))
    print(f"Appended {len(texts)} rows from {excel_path} into {db_path}")


if __name__ == "__main__":
    main()
