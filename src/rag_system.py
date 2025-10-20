"""RAG System Implementation"""

from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer


class RAGSystem:
    """A simple RAG system using ChromaDB for vector storage"""

    def __init__(self, collection_name: str = "documents", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG system

        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the sentence transformer model
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer(model_name)

    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store

        Args:
            documents: List of document texts
            metadata: Optional list of metadata dictionaries
        """
        if metadata is None:
            metadata = [{"index": i} for i in range(len(documents))]

        embeddings = self.model.encode(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store

        Args:
            query_text: The query string
            n_results: Number of results to return

        Returns:
            Dictionary containing the query results
        """
        query_embedding = self.model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results

    def clear(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
