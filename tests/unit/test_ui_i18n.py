"""Unit tests for the Gradio frontend's language catalog."""

from __future__ import annotations

import dataclasses

import pytest

from scinanoai.chatbot.ui_i18n import (
    DEFAULT_LANG,
    STRINGS,
    get_strings,
    other_lang,
    toggle_label,
)


@pytest.mark.unit
def test_default_language_is_registered() -> None:
    assert DEFAULT_LANG in STRINGS


@pytest.mark.unit
def test_every_language_defines_every_field() -> None:
    """No language may ship with a blank string (would render an empty UI label)."""
    fields = [f.name for f in dataclasses.fields(next(iter(STRINGS.values())))]
    for lang, strings in STRINGS.items():
        for field in fields:
            value = getattr(strings, field)
            assert isinstance(value, str) and value, f"{lang}.{field} is empty"


@pytest.mark.unit
def test_other_lang_round_trips() -> None:
    assert other_lang("ru") == "en"
    assert other_lang("en") == "ru"
    # Switching twice returns to the original language.
    assert other_lang(other_lang("ru")) == "ru"


@pytest.mark.unit
def test_get_strings_falls_back_to_default_for_unknown_language() -> None:
    assert get_strings("klingon").code == DEFAULT_LANG


@pytest.mark.unit
def test_toggle_label_advertises_the_target_language() -> None:
    """The button shows the language you'd switch *to*, not the current one."""
    assert toggle_label("ru") == "🌐 EN"
    assert toggle_label("en") == "🌐 RU"
