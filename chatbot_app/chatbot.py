import os
import tiktoken
import requests
from openai import OpenAI
from chatbot_app.conductor_agent import ConductorAgent

class ChatBot:
    def __init__(self, llm_model="openai/gpt-4o-mini"):
        self.llm_model=llm_model
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_api_base = os.environ["OPENAI_API_BASE"]
        self.conductor_agent = ConductorAgent(threshold=0.2)
        self.conversation_history = []

    def get_relevant_documents(self, query, k=10):
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
            # documents = self.get_relevant_documents(question)
            # context = "\n\n".join(doc['content'] for doc in documents)
            # prompt = (
            #     f"У вас есть следующая контекстная информация: {context} "
            #     f"Используя эту информацию, ответьте на следующий вопрос ясно, подробно и профессионально: "
            #     f"Вопрос: {question} Если информации в контексте недостаточно, признайте это честно "
            #     "и предложите логичный следующий шаг. Отвечайте на русском языке. Если для какого-то слова на русском нет аналога, то пиши это словно на английском"
            #     "Кроме того, если видишь специфичную терминологию с припиской нано, то оставляй ее на английском. Например: слово nanopillars -- его переводить нельзя"
            # )

            documents = self.get_relevant_documents(question)
            context_parts = []
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                filename = metadata.get("filename", "неизвестный_файл")
                context_parts.append(f"{content} [{filename}]")

            context = "\n\n".join(context_parts)
            
            prompt = (
                f"У вас есть следующая контекстная информация:\n\n"
                f"{context}\n\n"
                "Используя эту информацию, ответьте на следующий вопрос ясно, подробно и профессионально:\n\n"
                f"Вопрос: {question}\n\n"
                "Если информации в контексте недостаточно, признайте это честно и предложите логичный следующий шаг. "
                "Отвечайте на русском языке. Если для какого-то слова на русском нет аналога, то пишите это слово на английском. "
                "Кроме того, если видите специфичную терминологию с припиской нано, то оставляйте её на английском (например: nanopillars). "
                "\n\n"
                "Внимание: если используете информацию из конкретного фрагмента, приведите ссылку на название файла в квадратных скобках, "
                "как показано в контексте."
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
            model=self.llm_model,
            messages=messages,
            temperature=0,
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
        for message in messages:
            role = "Пользователь" if message['role'] == 'user' else "Ассистент"
            conversation += f"{role}: {message['content']}\n"

        prompt = (
            "Кратко суммируйте следующий диалог между пользователем и ассистентом, сохраняя важные детали. "
            "Отвечайте на русском языке.\n\n" + conversation
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        openai_client = OpenAI(base_url=self.openai_api_base)

        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            max_tokens=4096,
            temperature=0.2,
        )
        summary = response.choices[0].message.content.strip()
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
