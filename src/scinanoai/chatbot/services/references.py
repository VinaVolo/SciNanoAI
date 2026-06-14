"""Resolve reference filenames embedded in assistant answers to public links."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ...utils.paths import get_project_root

_BRACKET_RE = re.compile(r"\[(.*?)\]")


@lru_cache(maxsize=1)
def _load_reference_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["filename", "link_name"])
    return pd.read_csv(path)


class ReferenceResolver:
    """Replaces ``[filename.pdf]`` citations with their public URL counterparts."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self._csv_path = csv_path or get_project_root() / "data" / "updated_references_links.csv"

    def enrich(self, answer: str) -> str:
        table = _load_reference_table(self._csv_path)
        if table.empty:
            return answer

        rewritten = answer
        for citation in _BRACKET_RE.findall(answer):
            for filename in table["filename"].values:
                if citation in str(filename):
                    link = table[table["filename"] == filename].iloc[0]["link_name"]
                    rewritten = rewritten.replace(citation, str(link))
                    break
        return rewritten
