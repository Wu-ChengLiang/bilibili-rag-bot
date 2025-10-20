"""RAG Client with ChromaDB and Text2Vec"""

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from text2vec import SentenceModel
from .types import SearchResult


class RAGClient:
    """RAG Client with persistent ChromaDB storage and Text2Vec embeddings"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        model_name: str = "shibing624/text2vec-base-chinese"
    ):
        """Initialize RAG client

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            model_name: Text2Vec model name for Chinese embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize Text2Vec embedding model
        self.embedding_model = SentenceModel(model_name)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Text2Vec

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """Add a single document

        Args:
            content: Document content
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Document ID

        Raises:
            ValueError: If content is empty
        """
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty")

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        embedding = self._generate_embedding(content)

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata] if metadata else None
        )

        return doc_id

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add multiple documents

        Args:
            documents: List of document contents
            metadatas: Optional list of metadata dicts
            doc_ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]

        embeddings = [self._generate_embedding(doc) for doc in documents]

        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return doc_ids

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Optional minimum similarity score

        Returns:
            List of search results

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self._generate_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
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
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    "score": score
                }
                search_results.append(result)

        return search_results
