import os
from fastapi import FastAPI
from vector_db import VectorDatabase
from models import QueryRequest, QueryResponse, Document

app = FastAPI()

vector_db = VectorDatabase(
    db_path="/home/vinavolo/Documents/SciNanoAI/db/intfloat_multilingual-e5-large",
    #os.path.join("db", "intfloat_multilingual-e5-large"),
    model_name='intfloat/multilingual-e5-large'
)

@app.post("/query", response_model=QueryResponse)
def query_vector_db(request: QueryRequest):
    documents = vector_db.query(request.query, request.k, request.lambda_mult, request.fetch_k)
    response_documents = [
        Document(content=doc.page_content, metadata=doc.metadata) for doc in documents
    ]
    return QueryResponse(documents=response_documents)

@app.get("/num_documents")
def get_num_documents():
    num_documents = vector_db.get_num_documents()
    return {"num_documents": num_documents}
