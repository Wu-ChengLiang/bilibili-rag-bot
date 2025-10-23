"""Real-time Feishu RAG Client with daily vector store refresh"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.rag.client import RAGClient
from src.data.loaders import FeishuDocxLoader
from src.data.config import FeishuConfig
from src.rag.types import SearchResult

logger = logging.getLogger(__name__)


class RealTimeFeishuRAG:
    """
    Real-time Feishu RAG system with automatic daily updates.

    Features:
    - Fast search using cached vector store (ChromaDB)
    - Automatic daily refresh at midnight
    - Real-time document updates via scheduled tasks
    """

    def __init__(
        self,
        feishu_config: Optional[FeishuConfig] = None,
        doc_ids: Optional[List[str]] = None,
        rag_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RealTimeFeishuRAG

        Args:
            feishu_config: Feishu configuration (if None, load from .env)
            doc_ids: List of Feishu document IDs to load
            rag_config: RAG configuration (if None, use defaults)
        """
        # Load Feishu config
        if feishu_config is None:
            feishu_config = FeishuConfig.from_env()
        self.feishu_config = feishu_config

        # Document IDs to track
        self.doc_ids = doc_ids or []
        if not self.doc_ids:
            # If not specified, use wiki_space_id as default
            if self.feishu_config.wiki_space_id:
                self.doc_ids = [self.feishu_config.wiki_space_id]

        # Initialize RAG client
        self.rag_client = RAGClient(config=None)  # Use default RAG config

        # Initialize loader
        self.feishu_loader = FeishuDocxLoader(
            config=feishu_config,
            document_ids=self.doc_ids,
        )

        # Track update time
        self.last_update_time = None

        # Initialize vector store on startup
        logger.info("Initializing vector store from Feishu documents...")
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize or refresh vector store with Feishu documents"""
        try:
            logger.info(f"Loading {len(self.doc_ids)} Feishu document(s)...")

            # Load documents from Feishu
            documents = self.feishu_loader.load()

            if not documents:
                logger.warning("No documents loaded from Feishu")
                return

            logger.info(f"Loaded {len(documents)} document(s)")

            # Reset RAG client (clear old data)
            self.rag_client.reset()
            logger.info("Vector store cleared")

            # Add documents to vector store
            for doc in documents:
                metadata = {
                    "source": "feishu",
                    "doc_id": doc.doc_id,
                    "title": doc.title or "Untitled",
                    "url": doc.url or "",
                }
                metadata.update(doc.metadata or {})

                self.rag_client.add_document(
                    content=doc.content,
                    metadata=metadata,
                    doc_id=doc.doc_id,
                )

                logger.info(f"Added document: {doc.title} ({len(doc.content)} bytes)")

            # Update timestamp
            self.last_update_time = datetime.now()
            logger.info(f"✅ Vector store initialized at {self.last_update_time}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}", exc_info=True)
            raise

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search the vector store using cached data.

        This is a fast operation that uses the pre-loaded vector store.
        Vector store is updated daily via refresh_vector_store().

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{query}'")

        try:
            results = self.rag_client.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
            )

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def refresh_vector_store(self) -> bool:
        """
        Refresh vector store with latest Feishu documents.

        This should be called by the scheduler at midnight (00:00).
        It will reload documents from Feishu and update the vector store.

        Returns:
            True if refresh successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("🔄 Starting vector store refresh...")
        logger.info("=" * 60)

        try:
            self._initialize_vector_store()
            logger.info("✅ Vector store refresh completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Vector store refresh failed: {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        rag_stats = self.rag_client.get_stats()

        return {
            **rag_stats,
            "doc_count": len(self.doc_ids),
            "last_update_time": self.last_update_time.isoformat()
            if self.last_update_time
            else None,
            "system": "RealTimeFeishuRAG",
        }

    def __repr__(self) -> str:
        return (
            f"RealTimeFeishuRAG("
            f"docs={len(self.doc_ids)}, "
            f"last_update={self.last_update_time})"
        )
