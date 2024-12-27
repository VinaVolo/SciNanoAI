from pydantic import BaseModel
from typing import List, Dict

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    lambda_mult: float = 0.45 
    fetch_k: int = 50

class Document(BaseModel):
    content: str
    metadata: Dict

class QueryResponse(BaseModel):
    documents: List[Document]
