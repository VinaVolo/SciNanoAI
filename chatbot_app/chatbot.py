import os
import tiktoken
import requests
from openai import OpenAI
from conductor_agent import ConductorAgent

class ChatBot:
    def __init__(self):
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_api_base = os.environ["OPENAI_API_BASE"]
        self.conductor_agent = ConductorAgent(threshold=0.2)
        self.conversation_history = []

    def get_relevant_documents(self, query, k=5):
        url = "http://localhost:8000/query"
        payload = {"query": query, "k": k}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('documents', [])
        else:
            raise Exception(f"Ошибка при запросе к векторной базе данных: {response.text}")

    def generate_response(self, question):

        self.conversation_history.append({"role": "user", "content": question})

        total_tokens = self.count_tokens(self.conversation_history)
        max_model_tokens = 4096
        max_reply_tokens = 1000
        max_allowed_tokens = max_model_tokens - max_reply_tokens - 100
        
        if total_tokens > max_allowed_tokens:
            N = 4
            messages_to_summarize = self.conversation_history[:-N]
            recent_messages = self.conversation_history[-N:]

            summary = self.summarize_messages(messages_to_summarize)

            summary_message = {"role": "system", "content": f"Резюме предыдущего разговора: {summary}"}
            self.conversation_history = [summary_message] + recent_messages
            
        if self.conductor_agent.should_use_database(question):
            print("Условие для использования базы данных выполнено")
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
            
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        messages = self.limit_tokens(messages, max_tokens=max_allowed_tokens)
            
        openai_client = OpenAI(base_url=self.openai_api_base)

        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        
        reply = response.choices[0].message.content.strip()
        self.conversation_history.append({"role": "assistant", "content": reply})
        
        return reply

    def summarize_messages(self, messages):
        """
        Суммирует список сообщений с помощью OpenAI API.
        """
        # Формируем текст для суммирования
        conversation = ""
        print(messages)
        print("Начало суммирования")
        for message in messages:
            role = "Пользователь" if message['role'] == 'user' else "Ассистент"
            conversation += f"{role}: {message['content']}\n"

        prompt = (
            "Кратко суммируйте следующий диалог между пользователем и ассистентом, сохраняя важные детали. "
            "Отвечайте на русском языке.\n\n" + conversation
        )
        
        messages = {"role": "user", "content": prompt}
        
        print(messages)
        
        openai_client = OpenAI(base_url=self.openai_api_base)

        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            max_tokens=4096,
            temperature=0.2,
        )
        print("-------------------------------------------------------------------------------------------")
        summary = response.choices[0].message.content.strip()
        print(summary)
        return summary
    
    def count_tokens(self, messages):
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoding.encode(message['content']))
        return total_tokens

    def limit_tokens(self, messages, max_tokens):
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        total_tokens = 0
        limited_messages = []
        # Проходим сообщения с конца, чтобы сохранить последние сообщения
        for message in reversed(messages):
            message_tokens = len(encoding.encode(message['content']))
            if total_tokens + message_tokens > max_tokens:
                break
            limited_messages.insert(0, message)
            total_tokens += message_tokens
        return limited_messages
