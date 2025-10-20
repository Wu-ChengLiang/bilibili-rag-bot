"""Factory for creating vector store instances"""

from typing import Literal
from ..core.vector_store import BaseVectorStore
from ..config import VectorStoreConfig
from .chromadb_store import ChromaDBStore


def create_vector_store(
    config: VectorStoreConfig | None = None,
    provider: Literal["chromadb"] | None = None,
    **kwargs
) -> BaseVectorStore:
    """Factory function to create vector store instances

    Args:
        config: VectorStoreConfig instance
        provider: Vector store provider name (if config not provided)
        **kwargs: Additional arguments for vector store initialization

    Returns:
        BaseVectorStore instance

    Raises:
        ValueError: If provider is unsupported

    Examples:
        >>> # Using config
        >>> config = VectorStoreConfig(provider="chromadb")
        >>> store = create_vector_store(config)

        >>> # Using provider directly
        >>> store = create_vector_store(provider="chromadb", persist_directory="./db")
    """
    # Use config if provided, otherwise create from kwargs
    if config is None:
        if provider is None:
            provider = "chromadb"
        config = VectorStoreConfig(provider=provider, **kwargs)

    # Create vector store based on provider
    if config.provider == "chromadb":
        return ChromaDBStore(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            allow_reset=config.allow_reset,
        )
    # Future: Add support for Qdrant, Weaviate, etc.
    # elif config.provider == "qdrant":
    #     return QdrantStore(...)
    else:
        raise ValueError(f"Unsupported vector store provider: {config.provider}")
