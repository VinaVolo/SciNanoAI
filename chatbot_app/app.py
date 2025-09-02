import os
import gradio as gr
import requests

DB_API_URL = "http://localhost:8000"  # API for data base
CHAT_API_URL = "http://localhost:8001"  # API for chat

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

    try:
        response = requests.post(f"{CHAT_API_URL}/chat", json={"message": user_input})
        if response.status_code == 200:
            data = response.json()
            reply = data['reply']
            history.append((user_input, reply))
        else:
            error_message = f"Error in the chatbot: {response.text}"
            history.append((user_input, error_message))
        return "", history
    except Exception as e:
        error_message = f"Error connecting to the chatbot: {str(e)}"
        history.append((user_input, error_message))
        return "", history

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
            error_message = f"API error during cleaning: {response.text}"
            return [(None, error_message)]
    except Exception as e:
        error_message = f"Error connecting to the chatbot: {str(e)}"
        return [(None, error_message)]

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Welcome to ChatBot!")
    gr.Markdown("Hi! This is a RAG-based chatbot. There are 301 documents in the database. Ask a question!")

    chatbot_widget = gr.Chatbot()
    message_input = gr.Textbox(placeholder="Enter your question here...")
    submit_button = gr.Button("Send")
    clear_button = gr.Button("Clear the chat")

    message_input.submit(chat, inputs=[message_input, chatbot_widget], outputs=[message_input, chatbot_widget])
    submit_button.click(chat, inputs=[message_input, chatbot_widget], outputs=[message_input, chatbot_widget])
    clear_button.click(clear_chat, outputs=[chatbot_widget])

demo.launch(
    auth=(os.environ["username"], os.environ["password"]),
    auth_message="Enter your username and password",
    server_port=8517,
    server_name="0.0.0.0",
    root_path="/scinanoai"
)