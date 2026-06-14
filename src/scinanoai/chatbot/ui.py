"""Gradio frontend for the chatbot service.

Layout + event wiring only. Network access lives in :mod:`ui_api` and all copy /
styling lives in :mod:`ui_theme`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import gradio as gr

from ..utils.logging import setup_logging
from .settings import ChatbotSettings
from .ui_api import ChatApiClient, ChatApiError
from .ui_theme import (
    CHATBOT_PLACEHOLDER,
    CUSTOM_CSS,
    FORCE_DARK_JS,
    LATEX_DELIMITERS,
    PLACEHOLDER,
    SUBTITLE,
    TITLE,
    build_theme,
    status_html,
)

_LOG = setup_logging("scinanoai.ui")

# Cleared value for the multimodal input after a successful submit.
_EMPTY_INPUT: dict[str, Any] = {"text": "", "files": []}
# Backend requires a non-empty message; supply one when only images are attached.
_IMAGE_ONLY_PROMPT = "Проанализируй приложенные изображения."
_PENDING_TITLE = "Обрабатываю запрос…"
_PENDING_BODY = "Идёт поиск по базе знаний и генерация ответа. Это может занять до минуты."


def _pending_message() -> dict[str, Any]:
    """An assistant placeholder that renders Gradio's animated 'pending' spinner."""
    return {
        "role": "assistant",
        "content": _PENDING_BODY,
        "metadata": {"title": _PENDING_TITLE, "status": "pending"},
    }


def _user_turn(text: str, files: list[Any]) -> list[dict[str, Any]]:
    """Build the chat-history entries for a user turn (images, then text)."""
    turn: list[dict[str, Any]] = [
        {"role": "user", "content": {"path": path}} for path in files
    ]
    if text:
        turn.append({"role": "user", "content": text})
    return turn


def build_app(chat_api_url: str | None = None, root_path: str = "") -> gr.Blocks:
    client = ChatApiClient(chat_api_url)
    logout_url = f"{root_path}/logout" if root_path else "/logout"

    def _render_header(online: bool) -> str:
        return (
            "<div class='app-header'>"
            f"<div class='brand'><span class='title'>{TITLE}</span>"
            f"<span class='subtitle'>{SUBTITLE}</span></div>"
            "<div class='actions'>"
            f"{status_html(online)}"
            f"<a class='logout-link' href='{logout_url}'>⏏ Выйти</a>"
            "</div></div>"
        )

    def _on_load() -> tuple[str, str]:
        """Mint a fresh per-session id and probe backend health on page load."""
        return str(uuid4()), _render_header(client.health())

    def _respond(
        message: dict[str, Any], history: list[dict[str, Any]], session_id: str
    ) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
        if not text and not files:
            gr.Warning("Введите вопрос или приложите изображение.")
            yield message, history
            return

        history = history + _user_turn(text, files)
        history.append(_pending_message())
        yield _EMPTY_INPUT, history  # immediate feedback before the blocking call

        try:
            reply = client.send(text or _IMAGE_ONLY_PROMPT, files, session_id)
            history[-1] = {"role": "assistant", "content": reply}
        except ChatApiError as exc:
            gr.Warning(str(exc))
            history[-1] = {"role": "assistant", "content": f"⚠ {exc}"}
        yield _EMPTY_INPUT, history

    def _clear(session_id: str) -> list[dict[str, Any]]:
        try:
            client.clear(session_id)
            gr.Info("История диалога очищена.")
        except ChatApiError as exc:
            gr.Warning(str(exc))
        return []

    # Gradio 6 moved theme/css/js from the Blocks constructor to launch(); they are
    # applied in main(). build_app() only assembles the layout.
    demo = gr.Blocks(title="SciNanoAI", fill_height=True)
    with demo:
        session_state = gr.State("")
        header = gr.HTML(_render_header(False))

        chatbot = gr.Chatbot(
            height=560,
            render_markdown=True,
            latex_delimiters=LATEX_DELIMITERS,
            show_label=False,
            placeholder=CHATBOT_PLACEHOLDER,
        )
        message_input = gr.MultimodalTextbox(
            placeholder=PLACEHOLDER,
            file_types=["image"],
            file_count="multiple",
            sources=["upload"],
            show_label=False,
            autofocus=True,
        )
        with gr.Row():
            new_chat = gr.Button(
                "🗑 Новый диалог", variant="secondary", size="sm", scale=0
            )

        message_input.submit(
            _respond,
            inputs=[message_input, chatbot, session_state],
            outputs=[message_input, chatbot],
        )
        new_chat.click(_clear, inputs=[session_state], outputs=[chatbot])
        demo.load(_on_load, outputs=[session_state, header])

    return demo


def main() -> None:
    settings = ChatbotSettings()
    if not settings.gradio_username or not settings.gradio_password:
        raise RuntimeError(
            "Missing GRADIO_USERNAME / GRADIO_PASSWORD. Populate .env (see .env.example)."
        )
    demo = build_app(root_path=settings.gradio_root_path)
    demo.queue()
    demo.launch(
        theme=build_theme(),
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
        auth=(settings.gradio_username, settings.gradio_password),
        auth_message="Введите логин и пароль",
        server_port=settings.gradio_port,
        server_name="0.0.0.0",
        root_path=settings.gradio_root_path,
    )


if __name__ == "__main__":
    main()
