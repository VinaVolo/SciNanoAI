"""Gradio frontend for the chatbot service.

Layout + event wiring only. Network access lives in :mod:`ui_api`, translatable
copy in :mod:`ui_i18n`, and styling in :mod:`ui_theme`.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any
from uuid import uuid4

import gradio as gr

from ..utils.logging import setup_logging
from .settings import ChatbotSettings
from .ui_api import ChatApiClient, ChatApiError
from .ui_i18n import (
    AUTH_MESSAGE,
    DEFAULT_LANG,
    Strings,
    get_strings,
    other_lang,
    toggle_label,
)
from .ui_theme import (
    CUSTOM_CSS,
    FORCE_DARK_JS,
    LATEX_DELIMITERS,
    TITLE,
    build_theme,
    status_html,
)

_LOG = setup_logging("scinanoai.ui")

# Cleared value for the multimodal input after a successful submit.
_EMPTY_INPUT: dict[str, Any] = {"text": "", "files": []}


def _pending_message(strings: Strings) -> dict[str, Any]:
    """An assistant placeholder that renders Gradio's animated 'pending' spinner."""
    return {
        "role": "assistant",
        "content": strings.pending_body,
        "metadata": {"title": strings.pending_title, "status": "pending"},
    }


def _user_turn(text: str, files: list[Any]) -> list[dict[str, Any]]:
    """Build the chat-history entries for a user turn (images, then text)."""
    turn: list[dict[str, Any]] = [{"role": "user", "content": {"path": path}} for path in files]
    if text:
        turn.append({"role": "user", "content": text})
    return turn


def _new_chat_label(strings: Strings) -> str:
    return f"🗑 {strings.new_chat}"


def _render_header(online: bool, strings: Strings, logout_url: str) -> str:
    """The whole header banner: brand, status badge, language toggle, logout.

    The language toggle is a plain link (``.lang-link``); a document-level click
    handler relays it to the hidden ``#lang-trigger`` button (see FORCE_DARK_JS).
    """
    label = strings.status_online if online else strings.status_offline
    return (
        "<div class='app-header'>"
        f"<div class='brand'><span class='title'>{TITLE}</span>"
        f"<span class='subtitle'>{strings.subtitle}</span></div>"
        "<div class='actions'>"
        f"{status_html(online, label)}"
        "<a class='lang-link' href='javascript:void(0)' title='Switch language' "
        "onclick=\"document.getElementById('lang-trigger').click()\">"
        f"{toggle_label(strings.code)}</a>"
        f"<a class='logout-link' href='{logout_url}'>⏏ {strings.logout}</a>"
        "</div></div>"
    )


def _on_load(client: ChatApiClient, logout_url: str) -> tuple[str, bool, str]:
    """Mint a fresh per-session id and probe backend health on page load."""
    online = client.health()
    return str(uuid4()), online, _render_header(online, get_strings(DEFAULT_LANG), logout_url)


def _toggle_lang(logout_url: str, current_lang: str, online: bool) -> tuple[Any, ...]:
    """Flip RU ⇄ EN and re-render every language-dependent surface in place.

    Done as a server-side event (not a page reload) so the open conversation is
    preserved across a language switch.
    """
    new_lang = other_lang(current_lang)
    strings = get_strings(new_lang)
    return (
        new_lang,
        _render_header(online, strings, logout_url),
        gr.update(placeholder=strings.placeholder),
        gr.update(placeholder=strings.chatbot_placeholder),
        gr.update(value=_new_chat_label(strings)),
    )


def _respond(
    client: ChatApiClient,
    message: dict[str, Any],
    history: list[dict[str, Any]],
    session_id: str,
    lang: str,
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    strings = get_strings(lang)
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        gr.Warning(strings.warn_empty)
        yield message, history
        return

    history = history + _user_turn(text, files)
    history.append(_pending_message(strings))
    yield _EMPTY_INPUT, history  # immediate feedback before the blocking call

    try:
        reply = client.send(text or strings.image_only_prompt, files, session_id)
        history[-1] = {"role": "assistant", "content": reply}
    except ChatApiError as exc:
        gr.Warning(str(exc))
        history[-1] = {"role": "assistant", "content": f"⚠ {exc}"}
    yield _EMPTY_INPUT, history


def _clear(client: ChatApiClient, session_id: str, lang: str) -> list[dict[str, Any]]:
    strings = get_strings(lang)
    try:
        client.clear(session_id)
        gr.Info(strings.info_cleared)
    except ChatApiError as exc:
        gr.Warning(str(exc))
    return []


def build_app(chat_api_url: str | None = None, root_path: str = "") -> gr.Blocks:
    client = ChatApiClient(chat_api_url)
    logout_url = f"{root_path}/logout" if root_path else "/logout"
    initial = get_strings(DEFAULT_LANG)

    # Gradio 6 moved theme/css/js from the Blocks constructor to launch(); they are
    # applied in main(). build_app() only assembles the layout.
    demo = gr.Blocks(title="SciNanoAI", fill_height=True)
    with demo:
        session_state = gr.State("")
        lang_state = gr.State(DEFAULT_LANG)
        # Last-known backend health, so a language toggle re-renders the status
        # badge without a fresh probe.
        online_state = gr.State(False)

        header = gr.HTML(_render_header(False, initial, logout_url))
        # Hidden control behind the in-header language link; clicked via JS.
        lang_trigger = gr.Button("toggle language", elem_id="lang-trigger")

        chatbot = gr.Chatbot(
            height=560,
            render_markdown=True,
            latex_delimiters=LATEX_DELIMITERS,
            show_label=False,
            placeholder=initial.chatbot_placeholder,
        )
        message_input = gr.MultimodalTextbox(
            placeholder=initial.placeholder,
            file_types=["image"],
            file_count="multiple",
            sources=["upload"],
            show_label=False,
            autofocus=True,
        )
        with gr.Row():
            new_chat = gr.Button(_new_chat_label(initial), variant="secondary", size="sm", scale=0)

        message_input.submit(
            partial(_respond, client),
            inputs=[message_input, chatbot, session_state, lang_state],
            outputs=[message_input, chatbot],
        )
        new_chat.click(
            partial(_clear, client), inputs=[session_state, lang_state], outputs=[chatbot]
        )
        lang_trigger.click(
            partial(_toggle_lang, logout_url),
            inputs=[lang_state, online_state],
            outputs=[lang_state, header, message_input, chatbot, new_chat],
        )
        demo.load(
            partial(_on_load, client, logout_url),
            outputs=[session_state, online_state, header],
        )

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
        auth_message=AUTH_MESSAGE,
        server_port=settings.gradio_port,
        server_name="0.0.0.0",
        root_path=settings.gradio_root_path,
    )


if __name__ == "__main__":
    main()
