from pydantic import BaseModel
from typing import List, Dict

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class Document(BaseModel):
    content: str
    metadata: Dict

class QueryResponse(BaseModel):
    documents: List[Document]
