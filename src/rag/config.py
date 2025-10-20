"""Configuration management for RAG system"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""

    provider: Literal["text2vec", "openai", "custom"] = "text2vec"
    model_name: str = "shibing624/text2vec-base-chinese"
    batch_size: int = 32

    # For OpenAI (future extension)
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores"""

    provider: Literal["chromadb", "qdrant", "custom"] = "chromadb"
    persist_directory: str = "./chroma_db"
    collection_name: str = "documents"
    allow_reset: bool = True

    # For Qdrant (future extension)
    url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class RAGConfig:
    """Main configuration for RAG system"""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    # Search parameters
    default_search_limit: int = 5
    default_score_threshold: Optional[float] = None

    # Reranking
    enable_reranking: bool = False
    rerank_top_k: int = 5

    @classmethod
    def default_chinese(cls) -> "RAGConfig":
        """Create default configuration optimized for Chinese text

        Returns:
            RAGConfig with Chinese-optimized settings
        """
        return cls(
            embedding=EmbeddingConfig(
                provider="text2vec",
                model_name="shibing624/text2vec-base-chinese"
            ),
            vector_store=VectorStoreConfig(
                provider="chromadb",
                persist_directory="./chroma_db",
                collection_name="documents"
            ),
            enable_reranking=True
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGConfig":
        """Create configuration from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            RAGConfig instance
        """
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
        vector_store_config = VectorStoreConfig(**config_dict.get("vector_store", {}))

        return cls(
            embedding=embedding_config,
            vector_store=vector_store_config,
            default_search_limit=config_dict.get("default_search_limit", 5),
            default_score_threshold=config_dict.get("default_score_threshold"),
            enable_reranking=config_dict.get("enable_reranking", False),
            rerank_top_k=config_dict.get("rerank_top_k", 5),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary

        Returns:
            Configuration as dictionary
        """
        return {
            "embedding": {
                "provider": self.embedding.provider,
                "model_name": self.embedding.model_name,
                "batch_size": self.embedding.batch_size,
            },
            "vector_store": {
                "provider": self.vector_store.provider,
                "persist_directory": self.vector_store.persist_directory,
                "collection_name": self.vector_store.collection_name,
                "allow_reset": self.vector_store.allow_reset,
            },
            "default_search_limit": self.default_search_limit,
            "default_score_threshold": self.default_score_threshold,
            "enable_reranking": self.enable_reranking,
            "rerank_top_k": self.rerank_top_k,
        }
