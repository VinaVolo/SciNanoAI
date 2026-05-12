"""Prompt templates kept in a single place, with stable contracts."""

from __future__ import annotations

from textwrap import dedent

RAG_ANSWER_PROMPT = dedent(
    """\
    You have been provided with the following contextual information:

    Context:
    {context}

    Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual,
    not implementation-level.

    Based on this information, provide a clear, coherent, and professional answer to the
    following question:

    Question: {question}

    Compose a continuous, cohesive narrative emphasising the key technical aspects. Avoid
    bullet lists unless strictly necessary. When referencing data from the context, include
    citations in square brackets using the file name provided (e.g. [file_name.pdf]). The
    square brackets must contain only the file name. Do not introduce links that are not
    explicitly mentioned in the context. If the context is insufficient, say so openly and
    propose logical next steps.
    """
)


PLAIN_ANSWER_PROMPT = dedent(
    """\
    Provide a clear, detailed, and professional answer to the following question:

    Question: {question}
    """
)


JUDGE_PROMPT = dedent(
    """\
    Assess whether the provided answer satisfies the requirements stated in the question.
    Reply with a single token: "Yes" or "No".

    Question: {question}
    Answer: {answer}

    Rules:
    - "No" if information is insufficient, the question is still open, or partially correct.
    - "Yes" only when the answer is fully accurate, complete, and relevant.
    - For any ambiguous case default to "No".
    """
)


REFORMULATE_PROMPT = dedent(
    """\
    The answer to the following question was insufficient. Reformulate the question while
    preserving its original meaning. Output only the revised question text.

    Question: {question}
    """
)


SUMMARIZE_PROMPT = dedent(
    """\
    Provide a concise summary of the following conversation between a user and an assistant.
    Retain all key facts and decisions; do not invent details.

    Conversation:
    {conversation}
    """
)


IMAGE_ANALYSIS_PROMPT = dedent(
    """\
    Ты научный ассистент. Пользователь задал формальный вопрос и предоставил изображения.
    Доступны только извлечённые метрики (средний радиус и площадь объектов, классификация
    ядро/цитоплазма). Единицы: радиус — мкм, площадь — мкм^2.

    Сформируй связный ответ на русском языке: кратко перечисли, какие изображения и
    классы обнаружены, укажи усреднённые метрики, сделай выводы в контексте вопроса и
    зафиксируй ограничения (анализ только по предоставленным числам, нет пиксельного
    доступа).

    Метрики:
    {metrics}

    Вопрос: {question}
    """
)


IMAGE_NOTE_TEMPLATE = (
    "\n\nUser also uploaded {count} images, but this question is being answered from text sources."
)


def render_rag_prompt(*, context: str, question: str) -> str:
    return RAG_ANSWER_PROMPT.format(context=context, question=question)


def render_plain_prompt(question: str) -> str:
    return PLAIN_ANSWER_PROMPT.format(question=question)


def render_judge_prompt(*, question: str, answer: str) -> str:
    return JUDGE_PROMPT.format(question=question, answer=answer)


def render_reformulate_prompt(question: str) -> str:
    return REFORMULATE_PROMPT.format(question=question)


def render_summarize_prompt(conversation: str) -> str:
    return SUMMARIZE_PROMPT.format(conversation=conversation)


def render_image_prompt(*, question: str, metrics: str) -> str:
    return IMAGE_ANALYSIS_PROMPT.format(question=question, metrics=metrics)
