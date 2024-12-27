import os
import sys
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(project_root)
from unstructured.staging.base import elements_from_json
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.utils.paths import get_project_path
from vector_service.vector_db import *
from chatbot_app.chatbot import *


if __name__ == "__main__":

    vec = VectorDatabase(
        db_path=os.path.join(get_project_path(), "db", "intfloat_multilingual-e5-large_22"),
        model_name="intfloat/multilingual-e5-large"
    )

    chatbot = ChatBot(llm_model="openai/gpt-4o-mini", data_base=vec)


    data = pd.read_excel(os.path.join(get_project_path(), 'data', 'new_valid_answer.xlsx'))[['Вопрос', 'Правильный ответ ']]
    response = []
    for i in range(len(data)):
        response.append(chatbot.generate_response(data["Вопрос"].iloc[i]))
    data['RAG_Ответ'] = response

    data.to_csv(f'openai_save_.csv', index=False)
