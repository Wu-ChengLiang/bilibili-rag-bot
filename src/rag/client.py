"""RAG Client with dependency injection and flexible architecture"""

from typing import List, Dict, Any, Optional, Union
from .core.embedding import BaseEmbedding
from .core.vector_store import BaseVectorStore
from .embeddings.factory import create_embedding
from .stores.factory import create_vector_store
from .config import RAGConfig, EmbeddingConfig, VectorStoreConfig
from .types import SearchResult
from .reranker import Reranker


class RAGClient:
    """RAG Client with flexible embedding and vector store backends

    This client uses dependency injection to allow flexible configuration
    of embedding models and vector stores. It follows SOLID principles
    and supports easy testing and extension.

    Examples:
        >>> # Simple usage with defaults (Chinese-optimized)
        >>> client = RAGClient()

        >>> # Custom configuration
        >>> config = RAGConfig.default_chinese()
        >>> client = RAGClient.from_config(config)

        >>> # Dependency injection for testing
        >>> embedding = Mock(spec=BaseEmbedding)
        >>> store = Mock(spec=BaseVectorStore)
        >>> client = RAGClient(embedding=embedding, vector_store=store)
    """

    def __init__(
        self,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        config: Optional[RAGConfig] = None,
        enable_reranking: bool = False,
    ):
        """Initialize RAG client with dependency injection

        Args:
            embedding: Embedding model instance (if None, uses config or default)
            vector_store: Vector store instance (if None, uses config or default)
            config: RAG configuration (if None, uses default Chinese config)
            enable_reranking: Whether to enable reranking
        """
        # Use config if provided, otherwise use default
        if config is None:
            config = RAGConfig.default_chinese()

        self.config = config

        # Initialize embedding model (dependency injection or factory)
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = self._create_embedding_from_config(config.embedding)

        # Initialize vector store (dependency injection or factory)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = create_vector_store(config.vector_store)

        # Initialize reranker if enabled
        # Use explicit parameter if provided, otherwise use config
        if enable_reranking:
            self.enable_reranking = True
        else:
            self.enable_reranking = config.enable_reranking

        if self.enable_reranking:
            self.reranker = Reranker()
        else:
            self.reranker = None

    @classmethod
    def from_config(cls, config: RAGConfig) -> "RAGClient":
        """Create RAG client from configuration

        Args:
            config: RAG configuration

        Returns:
            RAGClient instance
        """
        return cls(config=config)

    @classmethod
    def default_chinese(cls) -> "RAGClient":
        """Create RAG client with Chinese-optimized defaults

        Returns:
            RAGClient configured for Chinese text processing
        """
        return cls(config=RAGConfig.default_chinese())

    def _create_embedding_from_config(self, config: EmbeddingConfig) -> BaseEmbedding:
        """Create embedding model from configuration

        Args:
            config: Embedding configuration

        Returns:
            BaseEmbedding instance

        Raises:
            ValueError: If provider is unsupported
        """
        if config.provider in ("text2vec", "gte"):
            return create_embedding(provider=config.provider, model_name=config.model_name)
        # Future: Add support for OpenAI, Cohere, etc.
        # elif config.provider == "openai":
        #     return OpenAIEmbedding(api_key=config.api_key, model=config.model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")

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

        # Generate embedding using injected embedding model
        embedding = self.embedding.encode(content)
        if not isinstance(embedding[0], list):
            # Single embedding, wrap in list
            embedding = [embedding]

        # Add to vector store
        ids = self.vector_store.add_documents(
            documents=[content],
            embeddings=embedding,
            metadatas=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None
        )

        return ids[0]

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
        if not documents:
            return []

        # Generate embeddings using injected embedding model
        embeddings = self.embedding.encode(documents)

        # Ensure embeddings is a list of lists
        if embeddings and not isinstance(embeddings[0], list):
            embeddings = [embeddings]

        # Add to vector store
        return self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=doc_ids
        )

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents

        Args:
            query: Search query
            limit: Maximum number of results (uses config default if None)
            score_threshold: Optional minimum similarity score
            filter_metadata: Optional metadata filter

        Returns:
            List of search results

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Use config defaults if not specified
        if limit is None:
            limit = self.config.default_search_limit
        if score_threshold is None:
            score_threshold = self.config.default_score_threshold

        # Generate query embedding
        query_embedding = self.embedding.encode(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit if not self.enable_reranking else limit * 2,  # Get more for reranking
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )

        # Apply reranking if enabled
        if self.enable_reranking and self.reranker and results:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=limit
            )

        return results[:limit]

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs

        Args:
            ids: List of document IDs to delete
        """
        self.vector_store.delete_documents(ids)

    def reset(self) -> None:
        """Clear all documents from the store"""
        self.vector_store.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system

        Returns:
            Dictionary with statistics
        """
        collection_info = self.vector_store.get_collection_info()

        return {
            "embedding_model": self.embedding.model_name,
            "embedding_dimension": self.embedding.dimension,
            "vector_store": self.vector_store.__class__.__name__,
            "collection_name": self.vector_store.collection_name,
            "document_count": self.vector_store.document_count,
            "reranking_enabled": self.enable_reranking,
            **collection_info,
        }

    @property
    def collection_name(self) -> str:
        """Get collection name"""
        return self.vector_store.collection_name

    @property
    def document_count(self) -> int:
        """Get document count"""
        return self.vector_store.document_count
