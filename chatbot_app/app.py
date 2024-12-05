import os
import gradio as gr
from chatbot import ChatBot

chatbot_instance = ChatBot()

def chat(user_input, history):
    try:
        response = chatbot_instance.generate_response(user_input)
        history.append((user_input, response))
        return "", history
    except Exception as e:
        error_message = f"Ошибка: {str(e)}"
        history.append((user_input, error_message))
        return "", history
    
def clear_chat():
    chatbot_instance.conversation_history = []
    return []

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

demo.launch(auth = (os.environ["username"], os.environ["password"]),
            auth_message="Enter your username and password",
            server_port=8517,
            server_name="0.0.0.0",
            root_path="/scinanoai")
