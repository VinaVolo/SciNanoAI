import os
import sys
sys.path.append('..')
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from src.utils.paths import get_project_path

os.environ["OPENAI_API_KEY"] = "сюда"
os.environ["OPENAI_API_BASE"] = "сюда"

@st.cache_resource()
def load_vector_database(path=os.path.join(get_project_path(), "db", "db_BAAI_bge-m3")):
    """
    Loads a vector database stored in the given path.

    Args:
        path: The path to the vector database.

    Returns:
        A FAISS vector store object.
    """
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
    vector_store = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store



def get_rag_response(question, retriever):
    """
    Generates an answer using RAG and OpenAI API.
    """
    relevant_documents = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in relevant_documents)

    openai_client = OpenAI(base_url=os.environ["OPENAI_API_BASE"])

    response = openai_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a highly intelligent assistant who helps find and explain information."
            },
            {
                "role": "user",
                "content": (
                    f"You have access to the following contextual information: {context} "
                    f"Using this information, answer the following question clearly, in detail, and professionally: "
                    f"Question: {question} If the information in the context is insufficient, acknowledge it honestly "
                    "and suggest a logical next step. Answer in Russian."
                )
            },
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    return response.choices[0].message.content.strip()


st.set_page_config(page_title="RAG ChatBot", page_icon="🤖", layout="wide")

st.title("🤖 Добро пожаловать в ChatBot!")
st.write("Привет! Это чат-бот на основе поисково-дополненной генерации (RAG). Во мне 301 документа. Задайте любой вопрос, и я постараюсь вам помочь!")

retriever = load_vector_database().as_retriever(search_type="similarity", search_kwargs={"k": 5})

user_input = st.text_input("Введите ваш вопрос здесь:")

if user_input:
    with st.spinner("Генерация ответа..."):
        try:
            response = get_rag_response(user_input, retriever)
            st.write(response)
        except Exception as e:
            st.error(f"Ошибка при обработке запроса: {e}")
