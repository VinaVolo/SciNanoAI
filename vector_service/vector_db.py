from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDatabase:
    def __init__(self, db_path, model_name):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = self.load_vector_database()

    def load_vector_database(self):
        """
        Loads the vector database from the specified path using FAISS.

        This method initializes and returns the vector store by loading
        it from the provided database path. It uses the HuggingFace embeddings
        for vector representation and allows potentially unsafe deserialization.

        Returns:
            FAISS: The loaded vector store instance.
        """
        return FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def query(self, query_text, k=5, lambda_mult=0.45, fetch_k=50):
        """
        Queries the vector database using a given query text.

        This method takes a query text as the primary argument and
        uses the vector store to retrieve the most relevant documents.
        The search type is set to "mmr" (maximal marginal relevance) and
        the search parameters are set to k, lambda_mult, and fetch_k.
        The method returns the list of relevant documents.

        Args:
            query_text (str): The query text to search with.
            k (int, optional): The number of relevant documents to fetch. Defaults to 5.
            lambda_mult (float, optional): The lambda multiplier for the mmr search. Defaults to 0.45.
            fetch_k (int, optional): The number of documents to fetch from the index. Defaults to 50.

        Returns:
            List[dict]: A list of dictionaries containing the relevant documents.
        """
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'lambda_mult': lambda_mult, "fetch_k":fetch_k}
        )

        relevant_documents = retriever.get_relevant_documents(query_text)
        return relevant_documents

    def get_num_documents(self) -> int:
        """
        Returns the number of documents stored inside the FAISS docstore.

        Returns:
            int: Count of documents available in the underlying vector store.
        """
        return len(self.vector_store.docstore._dict)
    
    def get_document_by_name(self, doc_name: str):
        """
        Retrieves a document from the vector database by its name.

        This method searches through the documents stored in the FAISS index
        and returns the first document whose metadata contains the given name.

        Args:
            doc_name (str): The name of the document to retrieve.

        Returns:
            dict or None: The document if found, otherwise None.
        """
        # В FAISS документы хранятся внутри docstore
        for doc_id, doc in self.vector_store.docstore._dict.items():
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            if metadata.get("filename") == doc_name:
                return doc
        return None
    
    
    def delete_document_by_name(self, doc_name: str) -> bool:
        """
        Deletes a document from the vector database by its name.

        This method searches through the FAISS index for a document whose
        metadata contains the given name and removes it from the database.

        Args:
            doc_name (str): The name of the document to delete.

        Returns:
            bool: True if the document was found and deleted, False otherwise.
        """
        # Ищем документ по имени в метаданных
        for doc_id, doc in self.vector_store.docstore._dict.items():
            metadata = getattr(doc, "metadata", {})
            if metadata.get("filename") == doc_name:
                # Удаляем из FAISS по doc_id
                self.vector_store.delete([doc_id])
                return True
        return False
