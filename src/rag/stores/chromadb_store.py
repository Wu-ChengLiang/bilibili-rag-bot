"""ChromaDB vector store implementation"""

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ..core.vector_store import BaseVectorStore
from ..types import SearchResult


class ChromaDBStore(BaseVectorStore):
    """ChromaDB implementation of vector store

    This implementation provides persistent storage using ChromaDB
    with cosine similarity for vector search.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        allow_reset: bool = True,
    ):
        """Initialize ChromaDB vector store

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            allow_reset: Whether to allow resetting the database
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=allow_reset
            )
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

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
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) must have the same length"
            )

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Add to ChromaDB
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return ids

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
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filter_metadata
        )

        # Format results
        search_results: List[SearchResult] = []

        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                # Calculate similarity score (ChromaDB returns distance, convert to similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Convert distance to similarity score

                # Apply threshold if specified
                if score_threshold is not None and score < score_threshold:
                    continue

                result: SearchResult = {
                    "doc_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else None,
                    "score": score
                }
                search_results.append(result)

        return search_results

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs

        Args:
            ids: List of document IDs to delete
        """
        if ids:
            self._collection.delete(ids=ids)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection

        Returns:
            Dictionary with collection metadata
        """
        count = self._collection.count()
        return {
            "name": self._collection_name,
            "count": count,
            "persist_directory": self._persist_directory,
        }

    def reset(self) -> None:
        """Clear all documents from the store"""
        # Delete the collection and recreate it
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def collection_name(self) -> str:
        """Get the name of the current collection"""
        return self._collection_name

    @property
    def document_count(self) -> int:
        """Get the number of documents in the collection"""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self._client.delete_collection(self._collection_name)
