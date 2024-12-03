import os
import sys
import pandas as pd

sys.path.append('..')

from tqdm import tqdm

from unstructured.staging.base import elements_from_json
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer, util

os.environ["OPENAI_API_KEY"] = "sk-or-vv-5b09823d83c2c4452cf48f669396b65a77e3fcb1f19f9090a98be38b124622f6"
os.environ["OPENAI_API_BASE"] = "https://api.vsegpt.ru/v1/"

import pandas as pd

from src.utils.paths import get_project_path
from openai import OpenAI
import time


def load_vector_db(load_path=os.path.join(get_project_path(), "db", "db_BAAI_bge-m3")):
    vector_store = FAISS.load_local(
        load_path, HuggingFaceEmbeddings(model_name='BAAI/bge-m3'),
        allow_dangerous_deserialization=True
    )
    return vector_store


def get_rag_response(ques, retriever):
    """
    Generates an answer using RAG and OpenAI API.
    """
    retrieved_docs = retriever.get_relevant_documents(ques)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = OpenAI(base_url=os.environ["OPENAI_API_BASE"])

    response_big = llm.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a highly intelligent assistant who helps find and explain information."
            },
            {
                "role": "user",
                "content": f"You have access to the following contextual information: {context} Using this information, answer the following question clearly, in detail, and professionally: Question: {ques} If the information in the context is insufficient, acknowledge it honestly and suggest a logical next step. Answer in Russian."
            },
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    return response_big.choices[0].message.content.strip()


def sequential_process_rag_responses(data, retriever):
    """
    Process RAG responses sequentially with rate limiting.
    """
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing RAG responses"):
        try:
            data.at[idx, 'RAG_Ответ'] = get_rag_response(row['Вопрос'], retriever)
            time.sleep(1.1)  # Wait for 1.1 seconds to respect the rate limit
        except Exception as e:
            print(f"Error processing question {idx}: {e}")


def evaluate_similarity(data):
    similarities = []
    for idx, row in data.iterrows():
        reference = row['Правильный ответ']
        prediction = row['RAG_Ответ']
        embeddings = model.encode([reference, prediction], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        similarities.append(similarity)
    return similarities


if __name__ == "__main__":
    db = load_vector_db()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    data = pd.read_csv(os.path.join(get_project_path(), 'data', 'Answer_Promt.csv'))[['Вопрос', 'Правильный ответ']]
    sequential_process_rag_responses(data, retriever)
    data.to_csv(f'openai_save_temp_0_2.csv', index=False)
    data = data.dropna()
    model = SentenceTransformer('BAAI/bge-m3')
    data['Семантическая_Близость'] = evaluate_similarity(data)
    average_similarity = data['Семантическая_Близость'].dropna().mean()
    print(f"Средняя семантическая близость: {average_similarity:.2f}")
    data.to_csv(f'openai_save_temp_0_2_2.csv', index=False)
