from __future__ import annotations

from typing import Iterable, Mapping


def build_contextual_prompt(question: str, context: str) -> str:
    return (
        "Вам предоставлена следующая контекстная информация:\n\n"
        f"{context}\n\n"
        "На основании этой информации дайте ясный, связный и профессиональный ответ на следующий вопрос:\n\n"
        f"Вопрос: {question}\n\n"
        "Ваш ответ должен быть написан в свободной, но профессиональной форме с использованием всех технических "
        "деталей, представленных в контексте. Избегайте использования структурирования текста в виде списков или "
        "подзаголовков, кроме тех случаев, когда это необходимо для пояснения сложных технических деталей. "
        "Сосредоточьтесь на создании общего текста, в котором раскрываются ключевые технические аспекты. "
        "Если информации из контекста недостаточно, честно признайте это, но постарайтесь предложить логичные "
        "следующие шаги для решения вопроса. Используйте русский язык, а для специфической терминологии, особенно с "
        "приставкой 'нано', оставляйте английские термины (например, nanopillars).\n\n"
        "При использовании данных из контекста обязательно приводите ссылки в квадратных скобках. Ссылки должны быть "
        "оформлены в соответствии с предоставленной информацией: либо это название файла в квадратных скобках, либо "
        "ссылка, оформленная по ГОСТ, также в квадратных скобках. Используйте только те ссылки, которые явно указаны "
        "в контексте."
    )


def build_direct_prompt(question: str) -> str:
    return (
        "Ответьте на следующий вопрос ясно, подробно и профессионально. "
        "Используйте русский язык и приводите технические детали, если они известны.\n\n"
        f"Вопрос: {question}"
    )


def format_context_documents(documents: Iterable[Mapping[str, object]]) -> str:
    context_parts = []
    for doc in documents:
        content = doc.get("content") if isinstance(doc, Mapping) else None
        metadata = doc.get("metadata") if isinstance(doc, Mapping) else None
        if isinstance(metadata, Mapping):
            filename = metadata.get("filename", "неизвестный_файл")
        else:
            filename = "неизвестный_файл"
        if isinstance(content, dict):
            # Some vector DBs may store nested fields.
            content_text = content.get("text") or str(content)
        else:
            content_text = str(content)
        context_parts.append(f"{content_text} [{filename}]")
    return "\n\n".join(context_parts)
