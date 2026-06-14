"""Unit tests for the Gradio frontend's API client helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from scinanoai.chatbot.ui_api import encode_images


def _write_png(tmp_path: Path, name: str = "img.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


@pytest.mark.unit
def test_encode_images_accepts_string_paths(tmp_path: Path) -> None:
    """MultimodalTextbox hands over plain string paths."""
    path = _write_png(tmp_path)

    encoded = encode_images([str(path)])

    assert len(encoded) == 1
    assert encoded[0]["name"] == "img.png"
    assert base64.b64decode(encoded[0]["data"]) == path.read_bytes()


@pytest.mark.unit
def test_encode_images_accepts_dict_and_object(tmp_path: Path) -> None:
    path = _write_png(tmp_path, "from_obj.png")

    class _FileObj:
        def __init__(self, p: str) -> None:
            self.path = p

    encoded = encode_images([{"path": str(path)}, _FileObj(str(path))])

    assert [item["name"] for item in encoded] == ["from_obj.png", "from_obj.png"]


@pytest.mark.unit
def test_encode_images_skips_missing_file(tmp_path: Path) -> None:
    """An unreadable path is logged and skipped, not fatal."""
    encoded = encode_images([str(tmp_path / "nope.png")])

    assert encoded == []


@pytest.mark.unit
def test_encode_images_handles_none_and_empty() -> None:
    assert encode_images(None) == []
    assert encode_images([]) == []
