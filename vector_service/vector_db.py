from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorDatabase:
    def __init__(self, db_path, model_name):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = self.load_vector_database()

    def load_vector_database(self):
        return FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def query(self, query_text, k=5, lambda_mult=0.45, fetch_k=50):
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'lambda_mult': lambda_mult, "fetch_k":fetch_k}
        )

        relevant_documents = retriever.get_relevant_documents(query_text)
        return relevant_documents
