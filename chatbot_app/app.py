import os
import base64
import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

DB_API_URL = "http://localhost:8000"  # API for data base
CHAT_API_URL = "http://localhost:8001"  # API for chat


def encode_images(image_files):
    encoded_images = []
    for idx, file_obj in enumerate(image_files or [], start=1):
        try:
            path = None
            if isinstance(file_obj, dict):
                path = file_obj.get("name") or file_obj.get("path")
            else:
                path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None)
            if path is None:
                continue
            with open(path, "rb") as file_handle:
                encoded_images.append(
                    {
                        "data": base64.b64encode(file_handle.read()).decode("utf-8"),
                        "name": os.path.basename(path) or f"image_{idx}.png",
                    }
                )
        except Exception as exc:
            print(f"Failed to encode image {file_obj}: {exc}")
    return encoded_images


def chat(user_input, history, image_files):
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

    try:
        if history is None:
            history = []

        encoded_images = encode_images(image_files)
        payload = {"message": user_input, "images": encoded_images}
        response = requests.post(f"{CHAT_API_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            reply = data['reply']
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
        else:
            error_message = f"Error in the chatbot: {response.text}"
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": error_message})
        return "", history, None
    except Exception as e:
        error_message = f"Error connecting to the chatbot: {str(e)}"
        history = history or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": error_message})
        return "", history, None

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
            return [], None
        else:
            error_message = f"API error during cleaning: {response.text}"
            return [{"role": "assistant", "content": error_message}], None
    except Exception as e:
        error_message = f"Error connecting to the chatbot: {str(e)}"
        return [{"role": "assistant", "content": error_message}], None

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Welcome to ChatBot!")
    gr.Markdown("Hi! This is a RAG-based chatbot. There are 301 documents in the database. Ask a question!")

    chatbot_widget = gr.Chatbot()
    message_input = gr.Textbox(placeholder="Enter your question here...")
    image_input = gr.File(label="Upload images", file_types=["image"], file_count="multiple")
    submit_button = gr.Button("Send")
    clear_button = gr.Button("Clear the chat")

    message_input.submit(
        chat,
        inputs=[message_input, chatbot_widget, image_input],
        outputs=[message_input, chatbot_widget, image_input],
    )
    submit_button.click(
        chat,
        inputs=[message_input, chatbot_widget, image_input],
        outputs=[message_input, chatbot_widget, image_input],
    )
    clear_button.click(clear_chat, outputs=[chatbot_widget, image_input])

username = os.getenv("username")
password = os.getenv("password")

if not username or not password:
    raise RuntimeError(
        "Missing username/password in environment. "
        "Populate the .env file (see env-example) or export the variables before launching the UI."
    )

demo.launch(
    auth=(username, password),
    auth_message="Enter your username and password",
    server_port=8517,
    server_name="0.0.0.0",
    root_path="/scinanoai"
)
