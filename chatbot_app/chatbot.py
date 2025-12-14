import os
import re
import json
import sys
import base64
from io import BytesIO
from pathlib import Path
import math

import numpy as np
import tiktoken
import requests
import pandas as pd
import logging
from PIL import Image
from openai import OpenAI
try:
    from .decomposer_agent import DecomposerAgent
except ImportError:
    from decomposer_agent import DecomposerAgent
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.chat_models.yandex import ChatYandexGPT
from dotenv import load_dotenv
try:
    from cellpose import models as cellpose_models
except ImportError:
    cellpose_models = None
try:
    from .local_agents import (
        LocalLLMAgent,
        LocalFormalityAgent,
        LocalImageDecisionAgent,
        FormalImageAnswerAgent,
    )
except ImportError:
    from local_agents import (
        LocalLLMAgent,
        LocalFormalityAgent,
        LocalImageDecisionAgent,
        FormalImageAnswerAgent,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import get_project_path
load_dotenv()

LOG_FILE = PROJECT_ROOT / "loggers_everything.log"
LOGGER_NAME = "scinano_ai"

logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class ChatBot:
    def __init__(self, llm_model):
        """
        Initialize the chatbot instance.

        Args:
            llm_model (str): The name of the language model to use.

        Sets the following instance variables:
            llm_model (str): The name of the language model to use.
            openai_api_key (str): The OpenAI API key.
            openai_api_base (str): The OpenAI API base URL.
            yandex_api_key (str): The Yandex API key.
            yandex_api_base (str): The Yandex API base URL.
            sber_api_key (str): The Sberbank API key.
            decomposer_agent (DecomposerAgent): The decomposer agent instance.
            conversation_history (list): The conversation history.
        """
        self.llm_model = llm_model
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.openai_api_base = os.environ["OPENAI_API_BASE"]
        self.yandex_api_key = os.environ["YANDEX_API_KEY"]
        self.yandex_api_base = os.environ["YANDEX_API_BASE"]
        self.sber_api_key = os.environ['SBER_API_KEY']
        self.local_api_key = os.getenv("LOCAL_API_KEY")
        self.local_api_base = os.getenv("LOCAL_API_BASE")

        self.decomposer_agent = DecomposerAgent(threshold=0.2)
        self.conversation_history = []
        self.local_client = None
        self.formal_request_agent = None
        self.image_decision_agent = None
        self.formal_image_agent = None
        self.last_image_decision = None
        self.logger = logger
        self.cellpose_model = None

        if self.local_api_base and self.local_api_key:
            self.local_client = LocalLLMAgent(
                api_base=self.local_api_base,
                api_key=self.local_api_key,
            )
            self.formal_request_agent = LocalFormalityAgent(self.local_client)
            self.image_decision_agent = LocalImageDecisionAgent(self.local_client)
            self.formal_image_agent = FormalImageAnswerAgent(self.local_client)
            self.logger.info("Initialized local decomposer agents for images.")
        else:
            self.logger.warning(
                "LOCAL_API_BASE/LOCAL_API_KEY not set. Image uploads will be rejected."
            )
        self.logger.info("ChatBot initialized with llm_model=%s", llm_model)

        cellpose_path = PROJECT_ROOT / "models" / "cellpose_v0_1" / "cellpose_full_stream_filtred"
        if cellpose_models and cellpose_path.exists():
            try:
                self.cellpose_model = cellpose_models.CellposeModel(
                    gpu=False, pretrained_model=str(cellpose_path)
                )
                self.logger.info("Cellpose model loaded from %s", cellpose_path)
            except Exception as exc:
                self.logger.error("Failed to load Cellpose model: %s", exc)
        else:
            self.logger.warning("Cellpose model not available; using stub segmentation.")
    def get_relevant_documents(self, query, k=10, lambda_mult=0.45, fetch_k=50):

        """
        Queries the vector database using the given query text and parameters.

        Args:
            query (str): The query text to search with.
            k (int, optional): The number of relevant documents to fetch. Defaults to 10.
            lambda_mult (float, optional): The lambda multiplier for the mmr search. Defaults to 0.45.
            fetch_k (int, optional): The number of documents to fetch from the index. Defaults to 50.

        Returns:
            List[dict]: A list of dictionaries containing the relevant documents.
        """
        url = "http://localhost:8000/query"
        payload = {"query": query, "k": k, "lambda_mult": lambda_mult, "fetch_k": fetch_k}
        response = requests.post(url, json=payload)
        self.logger.debug(
            "Vector DB query status=%s len=%s", response.status_code, len(response.text)
        )
        if response.status_code == 200:
            return response.json().get('documents', [])
        else:
            raise Exception(f"Error when querying the vector database: {response.text}")

    def generate_response(self, question, images=None):
        """
        Generates a response to a given question.

        The response is generated by querying a vector database for relevant documents and
        then using a language model to generate a response based on the context provided by
        the documents. The context is summarized and provided to the language model as a
        system message to help it generate a better response.

        The language model is also provided with a prompt that includes the question and
        information about how to answer it.

        If the response is deemed incomplete or inaccurate, the method calls itself
        recursively to generate a new response.

        Args:
            question (str): The question to answer.

        Returns:
            str: The generated response.
        """
        images = images or []
        self.logger.info(
            "generate_response called. question_length=%d images=%d",
            len(question),
            len(images),
        )
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
            summary_message = {"role": "system", "content": f"Summary of the previous conversation: {summary}"}
            self.conversation_history = [summary_message] + recent_messages
            self.logger.info("Conversation summarized to stay within token budget.")
        else:
            self.logger.debug(
                "Token check passed: total=%d allowed=%d", total_tokens, max_allowed_tokens
            )

        use_database = self.decomposer_agent.should_use_database(question)
        self.logger.info("Decomposer suggested database=%s", use_database)
        image_descriptions = []
        image_mode = "no_images"

        if images:
            image_descriptions = self.describe_images(images)
            if not self.formal_request_agent or not self.image_decision_agent:
                self.logger.error("Image provided but local agents unavailable.")
                raise RuntimeError(
                    "Для работы с изображениями необходимо задать LOCAL_API_BASE и LOCAL_API_KEY."
                )
            if self._looks_like_image_question(question):
                image_mode = "image_analysis"
                use_database = False
                self.logger.info("Heuristic forced image_analysis based on question text.")
            elif self.formal_request_agent.is_formal(question):
                self.logger.info("Question classified as FORMAL with images.")
                decision = self.image_decision_agent.decide(
                    question, "\n".join(image_descriptions)
                )
                image_mode = decision
                self.logger.info("Image decision agent returned mode=%s", decision)
                if decision == "literature":
                    use_database = True
                elif decision == "image_analysis":
                    use_database = False
            else:
                image_mode = "informal"
                self.logger.info("Question classified as INFORMAL with images.")

        self.last_image_decision = image_mode
        self.logger.info(
            "Final routing: use_database=%s image_mode=%s", use_database, image_mode
        )
        self.logger.debug("Image descriptions: %s", image_descriptions)

        if use_database:
            try:
                documents = self.get_relevant_documents(question)
            except Exception as exc:
                self.logger.error("Vector DB query failed, falling back. %s", exc)
                documents = []
            context_parts = []
            for doc in documents:
                content = doc.get("content", {})
                metadata = doc.get("metadata", {})
                filename = metadata.get("filename", "unknown_file")
                context_parts.append(f"{content} [{filename}]")

            context = "\n\n".join(context_parts)
            self.logger.info("Retrieved %d documents from vector DB.", len(context_parts))

            prompt = (
                "You have been provided with the following contextual information:\n\n"
                f"Contest: {context}\n\n"
                "Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.\n\n"
                "Based on this information, provide a clear, coherent, and professional answer to the following question:\n\n"
                f"Question: {question}\n\n"
                "Your response should be composed in a free, professional style, incorporating all technical details from the context. Avoid structuring your text as lists or subheadings unless it is necessary to clarify complex technical details. Focus on crafting a continuous, cohesive narrative that emphasizes the key technical aspects.\n\n"
                "If there is insufficient information in the context, acknowledge this openly, but suggest logical next steps to address the issue.\n\n"
                "When referencing data from the context, include references in square brackets, formatted using the file name provided in the context (e.g., [file_name.pdf])!!! Adhere strictly to this referencing requirement under all circumstances!!! The square brackets should only contain the file name, nothing else!\n\n"
                "After providing your answer, briefly validate whether all key technical details from the context were addressed and state any information gaps or uncertainties.\n\n"
                "Only use links that are explicitly mentioned in the context! Do not introduce links from external sources!"
            )
            if images and image_mode != "image_analysis":
                prompt += (
                    f"\n\nПользователь дополнительно загрузил {len(images)} изображений, "
                    "но текущая задача решается текстовыми источниками."
                )
        elif image_mode == "image_analysis":
            metrics_text, metrics_struct = self.process_images(images)
            self.logger.info("Image analysis metrics computed: %s", metrics_struct)
            if self.formal_image_agent:
                reply = self.formal_image_agent.generate(question, metrics_text)
                self.conversation_history.append({"role": "assistant", "content": reply})
                self.logger.info("Formal image agent reply generated.")
                return reply
            else:
                self.logger.info("Formal image agent unavailable; falling back to text-only prompt.")
                prompt = self.build_image_prompt(question, image_descriptions)
        else:
            prompt = (
                "Provide a clear, detailed, and professional answer to the following question:\n\n"
                f"Question: {question}"
            )
            if images:
                prompt += (
                    f"\n\nПользователь загрузил {len(images)} изображений для справки, но они не требуют анализа."
                )

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        self.logger.debug("Prompt prepared, length=%d", len(prompt))
        
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
                    model="GigaChat-Pro"
                )   
            reply = llm_gigachat.invoke(prompt).content
            
        elif self.llm_model == "gpt-oss:latest":
            
            print(f"Model {self.llm_model} is used. ")  
            
            url = os.getenv("LOCAL_API_BASE")
            headers = {"Authorization": f"Bearer {os.getenv("LOCAL_API_KEY")}"}
            data = {
                "model": "gpt-oss:latest",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 4096
            }
            
            response = requests.post(url, headers=headers, json=data)
            reply = response.json().get('choices')[0].get('message').get('content')

        else:
            print(f"Model {self.llm_model} is used. ")    

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
        self.logger.info(
            "Reply generated. chars=%d image_mode=%s", len(reply), image_mode
        )

        if self.judge_answer(reply, question) == "No" or self.judge_answer(reply, question) == "No.":
            return self.handle_incomplete_answer(question)
        
        df = pd.read_csv(os.path.join(get_project_path(), "data", "updated_references_links.csv"))
        finde_links_in_answer = self.extract_bracket_content(reply)
        new_answer = reply
        for link in finde_links_in_answer:
            for i in df["filename"].values:
                if link in i:
                    new_link = df[df["filename"] == i].iloc[0].link_name
                    new_answer = new_answer.replace(link, new_link)
        self.logger.info("Final answer post-processed with reference replacements.")
                
        return new_answer


    def judge_answer(self, answer, question):
        """
        Evaluates the given answer to the given question.

        The evaluation is done by creating a prompt that asks a language model to judge the answer.
        The prompt provides the question and the answer and asks the model to respond with 'Yes' if the answer is complete, accurate, and relevant and 'No' if the answer is incomplete or inaccurate.

        Args:
            answer (str): The answer to be evaluated.
            question (str): The question that the answer is supposed to answer.

        Returns:
            str: The verdict of the language model, either 'Yes' or 'No'.
        """
        prompt = (
            "Assess whether the provided answer satisfies the requirements stated in the question. Ensure the answer is complete, accurate, and relevant to the question.\n\n"
            "Begin with a concise checklist (3-5 bullets) summarizing the criteria you will evaluate: (1) Completeness relative to the question, (2) Factual accuracy, (3) Relevance to the question, (4) Explicit coverage of all requirements, (5) Absence of ambiguity or open issues.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            "Instructions:\n\n"
            "- If the answer states that the information is insufficient, the question is still open, or the context does not provide an answer, reply with 'No'.\n\n"
            "- If the answer is fully accurate, complete, and relevant, reply with 'Yes'.\n\n"
            "- If the answer is partially correct, only somewhat complete or relevant, reply with 'No'.\n\n"
            "- For any ambiguous or borderline cases where it is unclear whether the criteria are fully met, default to 'No' to maintain strict compliance.\n\n"
            "After making your assessment, validate your choice in 1-2 lines by explicitly stating which checklist items are satisfied or not.\n\n"
            "The output format must be either “Yes” or “No”.\n\n"
            
        )

        url = os.getenv("LOCAL_API_BASE")
        headers = {"Authorization": f"Bearer {os.getenv("LOCAL_API_KEY")}"}
        data = {
            "model": "gpt-oss:latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 10
        }
        
        response = requests.post(url, headers=headers, json=data)
        verdict = response.json().get('choices')[0].get('message').get('content')

        return verdict

    def handle_incomplete_answer(self, question):
        """
        Handles incomplete answers by reformulating the question.

        When the answer is evaluated as incomplete, this method is called to generate a new question.
        The new question is created by asking a language model to reformulate the original question.
        The reformulated question is then passed to the generate_response method to generate a new answer.

        Args:
            question (str): The original question that received an incomplete answer.

        Returns:
            str: The answer to the reformulated question.
        """
        prompt = (
            f"The answer to the following question was insufficient. Reformulate it while preserving its original meaning::\n\n"
            f"Question: {question}\n\n"
            "Suggest a revised version of the question."
        )
        
        url = os.getenv("LOCAL_API_BASE")
        headers = {"Authorization": f"Bearer {os.getenv("LOCAL_API_KEY")}"}
        data = {
            "model": "gpt-oss:latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 100
        }
        
        response = requests.post(url, headers=headers, json=data)
        reformulated_question = response.json().get('choices')[0].get('message').get('content')
        
        return self.generate_response(reformulated_question)

    def summarize_messages(self, messages):
        """
        Summarizes a conversation between a user and the chatbot.

        This method takes a list of messages as input, where each message is a dictionary containing the role of the sender (user or assistant) and the content of the message.
        The method returns a string that summarizes the conversation, preserving important details.

        The summary is generated by asking a language model to summarize the conversation.
        The language model is provided with a prompt that includes the conversation and the instruction to summarize it.
        The summary is then extracted from the language model's response.

        Args:
            messages (list): A list of dictionaries, where each dictionary contains the role of the sender (user or assistant) and the content of the message.

        Returns:
            str: A string that summarizes the conversation, preserving important details.
        """
        conversation = ""
        for message in messages:
            role = "User" if message['role'] == 'user' else "Assistant"
            conversation += f"{role}: {message['content']}\n"

        prompt = (
            f"Begin with a concise checklist (3-7 bullets) describing the major elements or events in the conversation. Then, provide a concise summary of the following conversation between the user and the assistant, ensuring that all key details are retained: {conversation}.\n\n"
            "After generating the summary, validate that all significant points and facts have been included and correct the summary if any important detail is missing."
        )

        messages = [{"role": "user", "content": prompt}]

        url = os.getenv("LOCAL_API_BASE")
        headers = {"Authorization": f"Bearer {os.getenv("LOCAL_API_KEY")}"}
        data = {
            "model": "gpt-oss:latest",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096
        }
        
        response = requests.post(url, headers=headers, json=data)
        summary = response.json().get('choices')[0].get('message').get('content')
        
        return summary

    def count_tokens(self, messages):
        """
        Counts the total number of tokens in a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries with 'role' and 'content' keys, each containing a message from the conversation.

        Returns:
            int: The total number of tokens in the conversation.
        """
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoding.encode(message['content']))
        return total_tokens

    def limit_tokens(self, messages, max_tokens):
        """
        Limits the total number of tokens in a list of messages to a given maximum value.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries with 'role' and 'content' keys, each containing a message from the conversation.
            max_tokens (int): The maximum number of tokens allowed in the conversation.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with 'role' and 'content' keys, each containing a message from the conversation, limited to the given maximum number of tokens.
        """
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

    def extract_bracket_content(self, text: str):
        return re.findall(r'\[(.*?)\]', text)

    def describe_images(self, encoded_images):
        descriptions = []
        for idx, payload in enumerate(encoded_images, start=1):
            try:
                if isinstance(payload, dict):
                    raw = base64.b64decode(payload.get("data", ""))
                else:
                    raw = base64.b64decode(payload)
                with Image.open(BytesIO(raw)) as img:
                    kb_size = len(raw) / 1024
                    descriptions.append(
                        f"Изображение {idx}: формат {img.format}, размер {img.width}x{img.height} px, приблизительно {kb_size:.1f} КБ."
                    )
            except Exception as exc:
                descriptions.append(f"Изображение {idx}: не удалось обработать ({exc}).")
                self.logger.error("Failed to describe image %d: %s", idx, exc)
        self.logger.info("Prepared %d image descriptions.", len(descriptions))
        return descriptions

    def build_image_prompt(self, question, image_descriptions):
        description_text = "\n".join(image_descriptions) if image_descriptions else "Изображения не удалось описать."
        return (
            "Ты выступаешь в роли научного ассистента, отвечающего за анализ экспериментальных изображений.\n"
            "У тебя нет прямого доступа к пикселям, доступны только текстовые описания ниже. "
            "Опиши, какие измерения, шаги постобработки и критерии оценки стоит применить, исходя из метаданных и формулировки вопроса.\n"
            f"{description_text}\n\n"
            f"Вопрос: {question}\n\n"
            "Сформулируй структурированный ответ на русском языке: кратко сформулируй цель анализа, опиши предложенный рабочий процесс, "
            "укажи ограничения, связанные с отсутствием прямого доступа к изображению, и зафиксируй, какие дополнительные данные необходимы."
        )

    def classify_image_color(self, img: Image.Image, threshold: int = 40) -> str:
        arr = np.array(img.convert("RGB")).astype(float)
        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]
        brightness = R + G + B
        mask = brightness > threshold
        if np.sum(mask) == 0:
            return "нет ярких пикселей"
        yellow_score = np.mean((R[mask] + G[mask]) - B[mask])
        blue_score = np.mean(B[mask] - (R[mask] + G[mask]))
        return "ядро (желтый)" if yellow_score > blue_score else "цитоплазма (синий)"

    def segment_image(self, img: Image.Image, threshold: int = 40):
        """
        Runs Cellpose if available; otherwise falls back to a brightness-based stub.
        Returns list of {"area": float, "radius": float}.
        """
        arr = np.array(img.convert("RGB"))
        if self.cellpose_model:
            try:
                masks, flows, styles = self.cellpose_model.eval(
                    [arr], channels=[0, 0], progress=False
                )
                mask = masks[0] if isinstance(masks, list) else masks
                entries = []
                for label in np.unique(mask):
                    if label == 0:
                        continue
                    area = float(np.sum(mask == label))
                    radius = math.sqrt(area / math.pi)
                    # scale factors for neural net output
                    radius *= (100 / 155)
                    area *= (100 / 155) ** 2
                    entries.append({"area": area, "radius": radius})
                return entries
            except Exception as exc:
                self.logger.error("Cellpose segmentation failed: %s", exc)
        # stub fallback
        arr_gray = np.array(img.convert("L")).astype(float)
        mask = arr_gray > threshold
        area = float(np.sum(mask))
        if area == 0:
            return []
        radius = math.sqrt(area / math.pi)
        return [{"area": area, "radius": radius}]

    def process_images(self, encoded_images):
        metrics = []
        summaries = []
        records = []
        for idx, payload in enumerate(encoded_images, start=1):
            try:
                if isinstance(payload, dict):
                    raw = base64.b64decode(payload.get("data", ""))
                    name = payload.get("name") or f"image_{idx}.png"
                else:
                    raw = base64.b64decode(payload)
                    name = f"image_{idx}.png"
                with Image.open(BytesIO(raw)) as img:
                    classification = self.classify_image_color(img)
                    segments = self.segment_image(img)
                    for seg in segments:
                        records.append(
                            {
                                "image_index": idx,
                                "image_name": name,
                                "material_name": name,
                                "classification": classification,
                                "area": seg["area"],
                                "radius": seg["radius"],
                            }
                        )
                    if not segments:
                        records.append(
                            {
                                "image_index": idx,
                                "image_name": name,
                                "material_name": name,
                                "classification": classification,
                                "area": 0.0,
                                "radius": 0.0,
                            }
                        )
            except Exception as exc:
                msg = f"Изображение {idx}: не удалось обработать ({exc})."
                summaries.append(msg)
                self.logger.error(msg)
        if records:
            df = pd.DataFrame(records)
            df_clean = self.remove_outliers_iqr(df, group_col="material_name", value_col="radius")
            df_clean = self.remove_outliers_iqr(df_clean, group_col="material_name", value_col="area")
            for name, group in df_clean.groupby("image_name"):
                avg_area = group["area"].mean()
                avg_radius = group["radius"].mean()
                classification = group["classification"].iloc[0]
                entry = {
                    "image_name": name,
                    "classification": classification,
                    "avg_area": avg_area,
                    "avg_radius": avg_radius,
                    "segments": len(group),
                }
                metrics.append(entry)
                summaries.append(
                    f"{name}: класс={classification}, средняя площадь={avg_area:.2f}, "
                    f"средний радиус={avg_radius:.2f}, сегментов после очистки={len(group)}."
                )
        return "\n".join(summaries), metrics

    def remove_outliers_iqr(self, df, group_col="material_name", value_col="radius"):
        cleaned_parts = []
        for name, group in df.groupby(group_col):
            Q1 = group[value_col].quantile(0.25)
            Q3 = group[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cleaned_group = group[(group[value_col] >= lower) & (group[value_col] <= upper)]
            cleaned_parts.append(cleaned_group)
        return pd.concat(cleaned_parts, ignore_index=True)

    def _looks_like_image_question(self, question: str) -> bool:
        q = (question or "").lower()
        keywords = ["изображ", "картин", "фото", "image", "picture", "photo"]
        return any(k in q for k in keywords)
