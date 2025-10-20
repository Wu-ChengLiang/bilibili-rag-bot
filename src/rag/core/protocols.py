"""Protocol definitions for RAG system"""

from typing import Protocol, List, runtime_checkable


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions

    Any callable that takes a list of texts and returns embeddings
    can be used as an embedding function.
    """

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations"""

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict] | None = None,
        ids: List[str] | None = None,
    ) -> List[str]:
        """Add documents with embeddings to the store"""
        ...

    def query(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filter_metadata: dict | None = None,
    ) -> dict:
        """Query similar documents"""
        ...

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        ...
