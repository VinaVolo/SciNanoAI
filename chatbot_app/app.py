import streamlit as st
from chatbot import ChatBot

def main():
    st.set_page_config(page_title="RAG ChatBot", page_icon="🤖", layout="wide")
    chatbot = ChatBot()

    st.title("🤖 Добро пожаловать в ChatBot!")
    st.write(f"Привет! Это чат-бот на основе RAG. В базе данных 301 документов. Задайте вопрос!")

    user_input = st.text_input("Введите ваш вопрос здесь:")

    if user_input:
        with st.spinner("Генерация ответа..."):
            try:
                response = chatbot.generate_response(user_input)
                st.write(response)
            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
