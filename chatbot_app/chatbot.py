import os
import json
import tiktoken
import requests
import pandas as pd
from openai import OpenAI
from decomposer_agent import DecomposerAgent
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.chat_models.yandex import ChatYandexGPT

class ChatBot:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_api_base = os.environ["OPENAI_API_BASE"]
        self.yandex_api_key = os.environ["YANDEX_API_KEY"]
        self.yandex_api_base = os.environ["YANDEX_API_BASE"]
        self.sber_api_key = os.environ['SBER_API_KEY']

        self.decomposer_agent = DecomposerAgent(threshold=0.2)
        self.conversation_history = []

    def get_relevant_documents(self, query, k=10, lambda_mult=0.45, fetch_k=50):

        url = "http://localhost:8000/query"
        payload = {"query": query, "k": k, "lambda_mult": lambda_mult, "fetch_k": fetch_k}
        response = requests.post(url, json=payload)
        print(response)
        if response.status_code == 200:
            return response.json().get('documents', [])
        else:
            raise Exception(f"Ошибка при запросе к векторной базе данных: {response.text}")

        # doce = self.data_base.query(query)
        # return [{"content":doc.page_content, "metadata":doc.metadata} for doc in doce]

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

        if self.decomposer_agent.should_use_database(question):
            documents = self.get_relevant_documents(question)
            context_parts = []
            for doc in documents:
                content = doc.get("content", {})
                metadata = doc.get("metadata", {})
                filename = metadata.get("filename", "неизвестный_файл")
                context_parts.append(f"{content} [{filename}]")

            context = "\n\n".join(context_parts)

            prompt = (
                f"Вам предоставлена следующая контекстная информация:\n\n"
                f"{context}\n\n"
                "На основании этой информации дайте ясный, связный и профессиональный ответ на следующий вопрос:\n\n"
                f"Вопрос: {question}\n\n"
                "Ваш ответ должен быть написан в свободной, но профессиональной форме с использованием всех технических деталей, представленных в контексте. Избегайте использования структурирования текста в виде списков или подзаголовков, "
                "кроме тех случаев, когда это необходимо для пояснения сложных технических деталей. Сосредоточьтесь на создании общего текста, "
                "в котором раскрываются ключевые технические аспекты"
                "Если информации из контекста недостаточно, честно признайте это, но постарайтесь предложить логичные следующие шаги для решения вопроса. "
                "Используйте русский язык, а для специфической терминологии, особенно с приставкой 'нано', оставляйте английские термины (например, nanopillars)."
                "\n\n"
                "При использовании данных из контекста обязательно приводите ссылки в квадратных скобках. Ссылки должны быть оформлены в соответствии с предоставленной информацией: либо это название файла в квадратных скобках, либо ссылка, оформленная по ГОСТ, также в квадратных скобках."
                "Используйте только те ссылки, которые явно указаны в контексте."
            )

        else:
            prompt = (
                f"Ответьте на следующий вопрос ясно, подробно и профессионально: "
                f"Вопрос: {question} Отвечайте на русском языке."
            )

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        if self.llm_model == "YandexGPT4":
            llm_yandex_gpt = ChatYandexGPT(
                    api_key=self.yandex_api_key,
                    model_uri=self.yandex_api_base,
                    model_name="yandexgpt-32k",
                    temperature=0.2,
                    max_tokens=4096,
                    messages=messages
                )
            
            reply = llm_yandex_gpt.invoke(prompt).content

        elif self.llm_model == "GigaChat-Pro":
            llm_gigachat = GigaChat(
                    credentials=self.sber_api_key,
                    verify_ssl_certs=False,
                    temperature=0.2,
                    max_tokens=4096,
                    model="GigaChat-lite"
                )   
            reply = llm_gigachat.invoke(prompt).content

        else:
            print(f"Используется модель: {self.llm_model}")    

            messages = self.limit_tokens(messages, max_tokens=max_allowed_tokens)

            openai_client = OpenAI(base_url=self.openai_api_base)

            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )

            reply = response.choices[0].message.content.strip()

        self.conversation_history.append({"role": "assistant", "content": reply})

        if self.judge_answer(reply, question) == "нет." or self.judge_answer(reply, question) == "нет":
            return self.handle_incomplete_answer(question)
        
        return reply


    def judge_answer(self, answer, question):
        """
        Оценивает, удовлетворяет ли ответ условиям полноты и релевантности.
        """
        prompt = (
            f"Оцени следующий ответ на соответствие заданному вопросу. Ответ должен быть полным, точным и релевантным:\n\n"
            f"Вопрос: {question}\n\n"
            f"Ответ: {answer}\n\n"
            "Если в ответе указывается, что информация недостаточна или вопрос остаётся открытым или что контекст не содержит ответ на вопрос, ответь 'Нет'. "
            "Если ответ полностью удовлетворяет критериям точности, полноты и релевантности, ответь 'Да'."
        )

        openai_client = OpenAI(base_url=self.openai_api_base)
        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )

        verdict = response.choices[0].message.content.strip().lower()
        return verdict

    def handle_incomplete_answer(self, question):
        """
        Переформулирует или декомпозирует вопрос и выполняет повторный запрос.
        """
        prompt = (
            f"Ответ на следующий вопрос оказался недостаточным. Переформулируй его с сохранением смысла:\n\n"
            f"Вопрос: {question}\n\n"
            "Предложите новый вариант вопроса"
        )

        openai_client = OpenAI(base_url=self.openai_api_base)
        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.5,
        )

        reformulated_question = response.choices[0].message.content.strip()
        return self.generate_response(reformulated_question)

    def summarize_messages(self, messages):
        """
        Суммирует список сообщений с помощью OpenAI API.
        """
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
        for message in reversed(messages):
            message_tokens = len(encoding.encode(message['content']))
            if total_tokens + message_tokens > max_tokens:
                break
            limited_messages.insert(0, message)
            total_tokens += message_tokens
        return limited_messages
