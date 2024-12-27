import os
import gradio as gr
import requests

DB_API_URL = "http://localhost:8000"  # API для базы данных
CHAT_API_URL = "http://localhost:8001"  # API для чата

def chat(user_input, history):
    try:
        # Отправляем запрос в чат API
        response = requests.post(f"{CHAT_API_URL}/chat", json={"message": user_input})
        if response.status_code == 200:
            data = response.json()
            reply = data['reply']
            history.append((user_input, reply))
        else:
            error_message = f"Ошибка в чат-боте: {response.text}"
            history.append((user_input, error_message))
        return "", history
    except Exception as e:
        error_message = f"Ошибка соединения с чат-ботом: {str(e)}"
        history.append((user_input, error_message))
        return "", history

def clear_chat():
    try:
        # Очищаем историю чата
        response = requests.post(f"{CHAT_API_URL}/clear_history")
        if response.status_code == 200:
            return []
        else:
            error_message = f"Ошибка API при очистке: {response.text}"
            return [(None, error_message)]
    except Exception as e:
        error_message = f"Ошибка соединения с чат-ботом: {str(e)}"
        return [(None, error_message)]

# Настройка интерфейса Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Добро пожаловать в ChatBot!")
    gr.Markdown("Привет! Это чат-бот на основе RAG. В базе данных 301 документ. Задайте вопрос!")

    chatbot_widget = gr.Chatbot()
    message_input = gr.Textbox(placeholder="Введите ваш вопрос здесь...")
    submit_button = gr.Button("Отправить")
    clear_button = gr.Button("Очистить чат")

    # Связываем кнопки с функциями
    message_input.submit(chat, inputs=[message_input, chatbot_widget], outputs=[message_input, chatbot_widget])  # Нажатие Enter
    submit_button.click(chat, inputs=[message_input, chatbot_widget], outputs=[message_input, chatbot_widget])  # Кнопка отправки
    clear_button.click(clear_chat, outputs=[chatbot_widget])

demo.launch(
    auth=(os.environ["username"], os.environ["password"]),
    auth_message="Enter your username and password",
    server_port=8517,
    server_name="0.0.0.0",
    root_path="/scinanoai"
)