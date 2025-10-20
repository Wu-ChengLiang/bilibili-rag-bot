"""Abstract base class for vector stores"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..types import SearchResult


class BaseVectorStore(ABC):
    """Abstract base class for vector database implementations

    All vector store implementations should inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents with pre-computed embeddings to the store

        Args:
            documents: List of document contents
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of document IDs

        Raises:
            ValueError: If documents and embeddings lengths don't match
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using query embedding

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Optional minimum similarity score
            filter_metadata: Optional metadata filter

        Returns:
            List of search results

        Raises:
            ValueError: If query_embedding is invalid
        """
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs

        Args:
            ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection

        Returns:
            Dictionary with collection metadata (name, count, etc.)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all documents from the store"""
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Get the name of the current collection"""
        pass

    @property
    @abstractmethod
    def document_count(self) -> int:
        """Get the number of documents in the collection"""
        pass
