"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path of the SciNanoAI project root."""
    return Path(__file__).resolve().parents[3]


# Backwards-compat alias for the legacy ``src.utils.paths.get_project_path``.
def get_project_path() -> Path:
    return get_project_root()
