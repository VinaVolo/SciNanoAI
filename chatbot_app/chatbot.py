import os
import requests
from openai import OpenAI
from conductor_agent import ConductorAgent

class ChatBot:
    def __init__(self):
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_api_base = os.environ["OPENAI_API_BASE"]
        self.conductor_agent = ConductorAgent(threshold=0.2)

    def get_relevant_documents(self, query, k=5):
        url = "http://localhost:8000/query"
        payload = {"query": query, "k": k}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('documents', [])
        else:
            raise Exception(f"Ошибка при запросе к векторной базе данных: {response.text}")

    def generate_response(self, question):
        if self.conductor_agent.should_use_database(question):
            documents = self.get_relevant_documents(question)
            context = "\n\n".join(doc['content'] for doc in documents)
            prompt = (
                f"У вас есть следующая контекстная информация: {context} "
                f"Используя эту информацию, ответьте на следующий вопрос ясно, подробно и профессионально: "
                f"Вопрос: {question} Если информации в контексте недостаточно, признайте это честно "
                "и предложите логичный следующий шаг. Отвечайте на русском языке."
            )
        else:
            prompt = (
                f"Ответьте на следующий вопрос ясно, подробно и профессионально: "
                f"Вопрос: {question} Отвечайте на русском языке."
            )
            
        messages = [
            {
                "role": "system",
                "content": "Вы — высокоинтеллектуальный помощник, который помогает находить и объяснять информацию."
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        openai_client = OpenAI(base_url=self.openai_api_base)

        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=3000,
        )
        return response.choices[0].message.content.strip()
