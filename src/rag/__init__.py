"""RAG Module with flexible architecture

This module provides a complete RAG (Retrieval-Augmented Generation) system
with dependency injection, flexible configuration, and extensible components.

Quick Start:
    >>> from rag import RAGClient
    >>> client = RAGClient()  # Uses Chinese-optimized defaults
    >>> client.add_documents(["文档1", "文档2"])
    >>> results = client.search("查询")

Advanced Usage:
    >>> from rag import RAGClient, RAGConfig
    >>> config = RAGConfig.default_chinese()
    >>> client = RAGClient.from_config(config)
"""

# Core abstractions
from .core.embedding import BaseEmbedding
from .core.vector_store import BaseVectorStore

# Implementations
from .embeddings.text2vec import Text2VecEmbedding
from .stores.chromadb_store import ChromaDBStore

# Configuration
from .config import RAGConfig, EmbeddingConfig, VectorStoreConfig

# Main client
from .client import RAGClient

# Utilities
from .document_loader import DocumentLoader
from .reranker import Reranker
from .needle_test import NeedleTest
from .llm_client import LLMClient

# Types
from .types import Document, SearchResult

__version__ = "2.0.0"

__all__ = [
    # Core
    "RAGClient",
    "RAGConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    # Abstractions
    "BaseEmbedding",
    "BaseVectorStore",
    # Implementations
    "Text2VecEmbedding",
    "ChromaDBStore",
    # Utilities
    "DocumentLoader",
    "Reranker",
    "NeedleTest",
    "LLMClient",
    # Types
    "Document",
    "SearchResult",
]
