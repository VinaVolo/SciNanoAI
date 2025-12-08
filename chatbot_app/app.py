import os

import gradio as gr
import requests

try:  # Allow running as script from chatbot_app directory.
    from chatbot_app.config import ChatbotSettings
except ModuleNotFoundError:  # pragma: no cover - CLI usage only
    import importlib
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    ChatbotSettings = importlib.import_module("chatbot_app.config").ChatbotSettings  # type: ignore

settings = ChatbotSettings.from_env()
CHAT_API_URL = settings.chat_api_url


def _normalize_messages(conversation):
    """
    Ensures each chat message is a dict with role/content fields.
    """
    normalized = []
    for message in conversation or []:
        if isinstance(message, dict) and "role" in message and "content" in message:
            normalized.append({"role": message["role"], "content": message["content"]})
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            user, assistant = message
            normalized.append({"role": "user", "content": user})
            normalized.append({"role": "assistant", "content": assistant})
        else:
            normalized.append({"role": "assistant", "content": str(message)})
    return normalized


def chat(user_input, history):
    """
    Sends a message to the chat API and updates the conversation history.

    Args:
        user_input (str): The user's input message to be sent to the chat API.
        history (list): The list of tuples representing the conversation history,
                        where each tuple contains a user input and a chatbot reply.

    Returns:
        tuple: A tuple containing an empty string and the updated conversation history.
               If the chat API responds successfully, the chatbot's reply is added to the history.
               If an error occurs, an error message is appended to the history.

    Raises:
        Exception: If there is a connection error with the chat API.
    """

    history = _normalize_messages(history)

    if not user_input or not user_input.strip():
        return history

    try:
        response = requests.post(f"{CHAT_API_URL}/chat", json={"message": user_input})
        if response.status_code == 200:
            data = response.json()
            conversation = data.get("conversation_history", [])
            return _normalize_messages(conversation)
        else:
            error_message = f"Ошибка в чат-боте: {response.text}"
    except Exception as e:
        error_message = f"Ошибка соединения с чат-ботом: {str(e)}"

    conversation_fallback = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": error_message},
    ]
    return _normalize_messages(conversation_fallback)

def clear_chat():
    """
    Sends a request to clear the chat history via the chat API.

    Returns:
        list: An empty list if the chat history is successfully cleared.
        list: A list containing a tuple with an error message if the API call fails or an exception occurs.
    """

    try:
        response = requests.post(f"{CHAT_API_URL}/clear_history")
        if response.status_code == 200:
            return []
        else:
            error_message = f"Ошибка API при очистке: {response.text}"
    except Exception as e:
        error_message = f"Ошибка соединения с чат-ботом: {str(e)}"
    return _normalize_messages([{"role": "assistant", "content": error_message}])

GRADIO_USERNAME = os.getenv("GRADIO_USERNAME") or os.getenv("username")
GRADIO_PASSWORD = os.getenv("GRADIO_PASSWORD") or os.getenv("password")


def _reset_textbox():
    return ""


with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Добро пожаловать в ChatBot!")
    gr.Markdown("Привет! Это чат-бот на основе RAG. В базе данных 301 документ. Задайте вопрос!")

    chatbot_widget = gr.Chatbot()
    message_input = gr.Textbox(placeholder="Введите ваш вопрос здесь...")
    submit_button = gr.Button("Отправить")
    clear_button = gr.Button("Очистить чат")

    message_submit = message_input.submit(
        chat,
        inputs=[message_input, chatbot_widget],
        outputs=[chatbot_widget],
    )
    message_submit.then(_reset_textbox, inputs=None, outputs=[message_input], queue=False)

    button_submit = submit_button.click(
        chat,
        inputs=[message_input, chatbot_widget],
        outputs=[chatbot_widget],
    )
    button_submit.then(_reset_textbox, inputs=None, outputs=[message_input], queue=False)
    clear_button.click(clear_chat, outputs=[chatbot_widget])

demo.launch(
    auth=(GRADIO_USERNAME, GRADIO_PASSWORD) if GRADIO_USERNAME and GRADIO_PASSWORD else None,
    auth_message="Enter your username and password",
    server_port=8517,
    server_name="0.0.0.0",
    root_path="/scinanoai"
)
