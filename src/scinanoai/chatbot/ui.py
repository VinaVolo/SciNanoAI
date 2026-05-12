"""Gradio frontend for the chatbot service."""

from __future__ import annotations

import base64
import os
from typing import Any

import gradio as gr
import httpx

from ..utils.logging import setup_logging
from .settings import ChatbotSettings

_LOG = setup_logging("scinanoai.ui")


def _encode_images(image_files: list[Any] | None) -> list[dict[str, str]]:
    encoded: list[dict[str, str]] = []
    for idx, file_obj in enumerate(image_files or [], start=1):
        path = None
        if isinstance(file_obj, dict):
            path = file_obj.get("name") or file_obj.get("path")
        else:
            path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None)
        if path is None:
            continue
        try:
            with open(path, "rb") as fh:
                encoded.append(
                    {
                        "data": base64.b64encode(fh.read()).decode("utf-8"),
                        "name": os.path.basename(path) or f"image_{idx}.png",
                    }
                )
        except OSError as exc:
            _LOG.error("Failed to read image %s: %s", path, exc)
    return encoded


def build_app(chat_api_url: str | None = None) -> gr.Blocks:
    chat_api = chat_api_url or os.getenv("CHAT_API_URL", "http://localhost:8001")

    def _chat(user_input: str, history: list | None, image_files: list[Any] | None):
        history = history or []
        try:
            payload = {"message": user_input, "images": _encode_images(image_files)}
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{chat_api}/chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": data["reply"]})
            else:
                history.append({"role": "user", "content": user_input})
                history.append(
                    {"role": "assistant", "content": f"Error in the chatbot: {response.text}"}
                )
        except httpx.HTTPError as exc:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": f"Error connecting: {exc}"})
        return "", history, None

    def _clear():
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{chat_api}/clear_history")
            if response.status_code == 200:
                return [], None
            return [{"role": "assistant", "content": f"API error: {response.text}"}], None
        except httpx.HTTPError as exc:
            return [{"role": "assistant", "content": f"Connection error: {exc}"}], None

    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Welcome to SciNanoAI ChatBot!")
        gr.Markdown(
            "RAG-based chatbot over the SciNanoAI knowledge base. "
            "Ask a question or attach images for analysis."
        )

        chatbot_widget = gr.Chatbot(type="messages")
        message_input = gr.Textbox(placeholder="Enter your question here...")
        image_input = gr.File(label="Upload images", file_types=["image"], file_count="multiple")
        submit_button = gr.Button("Send")
        clear_button = gr.Button("Clear the chat")

        for trigger in (message_input.submit, submit_button.click):
            trigger(
                _chat,
                inputs=[message_input, chatbot_widget, image_input],
                outputs=[message_input, chatbot_widget, image_input],
            )
        clear_button.click(_clear, outputs=[chatbot_widget, image_input])

    return demo


def main() -> None:
    settings = ChatbotSettings()
    if not settings.gradio_username or not settings.gradio_password:
        raise RuntimeError(
            "Missing GRADIO_USERNAME / GRADIO_PASSWORD. Populate .env (see .env.example)."
        )
    demo = build_app()
    demo.launch(
        auth=(settings.gradio_username, settings.gradio_password),
        auth_message="Enter your username and password",
        server_port=settings.gradio_port,
        server_name="0.0.0.0",
        root_path=settings.gradio_root_path,
    )


if __name__ == "__main__":
    main()
