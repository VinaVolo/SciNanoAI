from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatRequest(BaseModel):
    message: str
    images: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    reply: str
    conversation_history: List[Dict[str, str]]
