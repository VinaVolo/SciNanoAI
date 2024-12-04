import os
from fastapi import FastAPI
from vector_db import VectorDatabase
from models import QueryRequest, QueryResponse, Document


app = FastAPI()

# Инициализация векторной базы данных при запуске сервиса
vector_db = VectorDatabase(
    db_path=os.path.join("db", "db_BAAI_bge-m3"),
    model_name='BAAI/bge-m3'
)

@app.post("/query", response_model=QueryResponse)
def query_vector_db(request: QueryRequest):
    documents = vector_db.query(request.query, request.k)
    response_documents = [
        Document(content=doc.page_content, metadata=doc.metadata) for doc in documents
    ]
    return QueryResponse(documents=response_documents)

@app.get("/num_documents")
def get_num_documents():
    num_documents = vector_db.get_num_documents()
    return {"num_documents": num_documents}
