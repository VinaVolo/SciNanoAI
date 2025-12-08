from __future__ import annotations

from typing import List

from ..llm.base import LLMClient, Message


class ConversationSummarizer:
    """
    Summarizes long-running conversations to keep token usage in check.
    """

    def __init__(self, llm_client: LLMClient):
        self._client = llm_client

    def summarize(self, messages: List[Message]) -> str:
        if not messages:
            return ""
        conversation = []
        for message in messages:
            role = "Пользователь" if message["role"] == "user" else "Ассистент"
            conversation.append(f"{role}: {message['content']}")
        prompt = (
            "Кратко суммируйте следующий диалог между пользователем и ассистентом, сохраняя важные детали. "
            "Отвечайте на русском языке.\n\n" + "\n".join(conversation)
        )
        return self._client.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )


class AnswerEvaluator:
    """
    Asks the LLM to verify whether the produced answer is complete and relevant.
    """

    def __init__(self, llm_client: LLMClient):
        self._client = llm_client

    def is_answer_complete(self, answer: str, question: str) -> bool:
        prompt = (
            "Оцени следующий ответ на соответствие заданному вопросу. Ответ должен быть полным, точным и релевантным:\n\n"
            f"Вопрос: {question}\n\n"
            f"Ответ: {answer}\n\n"
            "Если в ответе указывается, что информация недостаточна или вопрос остаётся открытым, или что контекст не "
            "содержит ответ на вопрос, ответь 'Нет'. Если ответ полностью удовлетворяет критериям точности, полноты и "
            "релевантности, ответь 'Да'."
        )
        verdict = self._client.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        return verdict.strip().lower().startswith("да")


class QuestionRephraser:
    """
    Reformulates the question when the initial answer quality is insufficient.
    """

    def __init__(self, llm_client: LLMClient):
        self._client = llm_client

    def rephrase(self, question: str) -> str:
        prompt = (
            "Ответ на следующий вопрос оказался недостаточным. Переформулируй его с сохранением смысла и предложи новый "
            "вариант:\n\n"
            f"Вопрос: {question}"
        )
        return self._client.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5,
        )
