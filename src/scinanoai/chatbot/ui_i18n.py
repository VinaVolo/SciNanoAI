"""User-facing copy for the Gradio frontend, in every supported UI language.

All translatable strings live here so :mod:`ui.py` stays layout + wiring and
:mod:`ui_theme.py` stays purely visual. Add a language by adding one entry to
:data:`STRINGS`; the header language toggle picks it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

# Brand wordmark is language-agnostic, so it lives with the theme, not here.

DEFAULT_LANG = "ru"


@dataclass(frozen=True)
class Strings:
    """Every translatable string for a single UI language."""

    code: str
    subtitle: str
    placeholder: str
    chatbot_placeholder: str
    new_chat: str
    logout: str
    status_online: str
    status_offline: str
    pending_title: str
    pending_body: str
    image_only_prompt: str
    warn_empty: str
    info_cleared: str


STRINGS: dict[str, Strings] = {
    "ru": Strings(
        code="ru",
        subtitle="Научный ассистент по базе знаний о наноструктурах",
        placeholder="Задайте вопрос или приложите изображение для анализа…",
        chatbot_placeholder=(
            "### 🔬 Чем помочь?\nСпросите про наноструктуры или приложите изображение для анализа."
        ),
        new_chat="Новый диалог",
        logout="Выйти",
        status_online="сервис на связи",
        status_offline="сервис недоступен",
        pending_title="Обрабатываю запрос…",
        pending_body=("Идёт поиск по базе знаний и генерация ответа. Это может занять до минуты."),
        image_only_prompt="Проанализируй приложенные изображения.",
        warn_empty="Введите вопрос или приложите изображение.",
        info_cleared="История диалога очищена.",
    ),
    "en": Strings(
        code="en",
        subtitle="Scientific assistant for the nanostructure knowledge base",
        placeholder="Ask a question or attach an image for analysis…",
        chatbot_placeholder=(
            "### 🔬 How can I help?\nAsk about nanostructures or attach an image for analysis."
        ),
        new_chat="New chat",
        logout="Log out",
        status_online="service online",
        status_offline="service offline",
        pending_title="Processing your request…",
        pending_body=(
            "Searching the knowledge base and generating an answer. This may take up to a minute."
        ),
        image_only_prompt="Analyze the attached images.",
        warn_empty="Enter a question or attach an image.",
        info_cleared="Conversation history cleared.",
    ),
}

# Bilingual so the login gate (shown before any language is chosen) reads in both.
AUTH_MESSAGE = "Введите логин и пароль · Enter your username and password"


def get_strings(lang: str) -> Strings:
    """Return the catalog for ``lang``, falling back to the default language."""
    return STRINGS.get(lang, STRINGS[DEFAULT_LANG])


def other_lang(lang: str) -> str:
    """The language the toggle switches to from ``lang`` (RU ⇄ EN)."""
    return "en" if get_strings(lang).code == "ru" else "ru"


def toggle_label(lang: str) -> str:
    """Label for the header toggle button: a globe plus the *target* language."""
    return f"🌐 {other_lang(lang).upper()}"
