import requests


class LocalLLMAgent:
    def __init__(self, api_base: str, api_key: str, model: str = "gpt-oss:latest"):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, temperature: float = 0, max_tokens: int = 256) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(self.api_base, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        choice = response.json()["choices"][0]["message"]["content"]
        return choice.strip()


class LocalFormalityAgent:
    def __init__(self, client: LocalLLMAgent):
        self.client = client

    def is_formal(self, question: str) -> bool:
        prompt = (
            "Определи, является ли запрос формальным научно-техническим вопросом, который требует строгого ответа.\n"
            "Ответь одним словом: 'formal' или 'informal'.\n"
            f"Запрос: {question}"
        )
        verdict = self.client.complete(prompt).lower()
        return verdict.startswith("formal")


class LocalImageDecisionAgent:
    def __init__(self, client: LocalLLMAgent):
        self.client = client

    def decide(self, question: str, image_notes: str) -> str:
        prompt = (
            "Пользователь задал формальный вопрос и загрузил изображения.\n"
            "Нужно определить дальнейший план:\n"
            "- Ответь 'image_analysis', если необходим детальный анализ изображений.\n"
            "- Ответь 'literature', если достаточно поиска литературы/текстовых источников.\n"
            f"Описание изображений:\n{image_notes}\n\n"
            f"Вопрос: {question}"
        )
        verdict = self.client.complete(prompt).lower()
        if verdict.startswith("image"):
            return "image_analysis"
        return "literature"


class FormalImageAnswerAgent:
    def __init__(self, client: LocalLLMAgent):
        self.client = client

    def generate(self, question: str, metrics_summary: str) -> str:
        prompt = (
            "Ты научный ассистент. Пользователь задал формальный вопрос и предоставил изображения.\n"
            "Доступны только извлеченные метрики по каждой картинке (средний радиус и площадь объектов, а также класс ядро/цитоплазма).\n"
            "Сформируй связный ответ на русском языке: кратко перечисли, какие изображения и классы обнаружены, укажи усредненные метрики, "
            "сделай выводы в контексте вопроса и зафиксируй ограничения (анализ только по предоставленным числам, нет пиксельного доступа).\n"
            f"Метрики:\n{metrics_summary}\n\n"
            f"Вопрос: {question}"
        )
        return self.client.complete(prompt, temperature=0.2, max_tokens=2048)
