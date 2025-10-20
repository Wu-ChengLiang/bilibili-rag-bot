"""Unit tests for refactored RAG client with dependency injection"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

# Import all necessary components
from src.rag import (
    RAGClient,
    RAGConfig,
    BaseEmbedding,
    BaseVectorStore,
    Text2VecEmbedding,
    ChromaDBStore,
)
from src.rag.types import SearchResult


class MockEmbedding(BaseEmbedding):
    """Mock embedding for testing"""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def encode(self, texts):
        """Generate fake embeddings"""
        if isinstance(texts, str):
            return [0.1] * self._dimension
        return [[0.1] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing"""

    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        self.metadatas = {}
        self._collection_name = "test_collection"

    def add_documents(self, documents, embeddings, metadatas=None, ids=None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        for i, doc_id in enumerate(ids):
            self.documents[doc_id] = documents[i]
            self.embeddings[doc_id] = embeddings[i]
            if metadatas:
                self.metadatas[doc_id] = metadatas[i]

        return ids

    def search(self, query_embedding, limit=5, score_threshold=None, filter_metadata=None):
        # Simple mock search - just return first N documents
        results = []
        for doc_id in list(self.documents.keys())[:limit]:
            result: SearchResult = {
                "doc_id": doc_id,
                "content": self.documents[doc_id],
                "metadata": self.metadatas.get(doc_id),
                "score": 0.9,
            }
            results.append(result)
        return results

    def delete_documents(self, ids):
        for doc_id in ids:
            self.documents.pop(doc_id, None)
            self.embeddings.pop(doc_id, None)
            self.metadatas.pop(doc_id, None)

    def get_collection_info(self):
        return {"name": self._collection_name, "count": len(self.documents)}

    def reset(self):
        self.documents.clear()
        self.embeddings.clear()
        self.metadatas.clear()

    @property
    def collection_name(self):
        return self._collection_name

    @property
    def document_count(self):
        return len(self.documents)


class TestDependencyInjection:
    """Test dependency injection works correctly"""

    def test_inject_custom_embedding(self):
        """Test that custom embedding can be injected"""
        mock_embedding = MockEmbedding(dimension=128)
        mock_store = MockVectorStore()

        client = RAGClient(embedding=mock_embedding, vector_store=mock_store)

        assert client.embedding is mock_embedding
        assert client.embedding.dimension == 128

    def test_inject_custom_vector_store(self):
        """Test that custom vector store can be injected"""
        mock_embedding = MockEmbedding()
        mock_store = MockVectorStore()

        client = RAGClient(embedding=mock_embedding, vector_store=mock_store)

        assert client.vector_store is mock_store
        assert client.collection_name == "test_collection"

    def test_uses_config_when_no_injection(self):
        """Test that client uses config when dependencies not injected"""
        config = RAGConfig.default_chinese()

        # This should work without errors even though we're not injecting
        # (it will create real instances)
        with patch('src.rag.client.Text2VecEmbedding') as mock_text2vec:
            with patch('src.rag.client.create_vector_store') as mock_factory:
                mock_text2vec.return_value = MockEmbedding()
                mock_factory.return_value = MockVectorStore()

                client = RAGClient.from_config(config)

                assert client.config == config
                mock_text2vec.assert_called_once()
                mock_factory.assert_called_once()


class TestRAGClientOperations:
    """Test RAG client operations with mocked dependencies"""

    @pytest.fixture
    def client(self):
        """Fixture providing a test client with mocks"""
        embedding = MockEmbedding()
        store = MockVectorStore()
        return RAGClient(embedding=embedding, vector_store=store)

    def test_add_single_document(self, client):
        """Test adding a single document"""
        doc_id = client.add_document("Test document", metadata={"source": "test"})

        assert doc_id is not None
        assert client.document_count == 1

    def test_add_multiple_documents(self, client):
        """Test adding multiple documents"""
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        ids = client.add_documents(docs)

        assert len(ids) == 3
        assert client.document_count == 3

    def test_search_documents(self, client):
        """Test searching documents"""
        # Add documents
        client.add_documents(["Doc 1", "Doc 2", "Doc 3"])

        # Search
        results = client.search("test query", limit=2)

        assert len(results) <= 2
        assert all(isinstance(r, dict) for r in results)
        assert all("doc_id" in r for r in results)

    def test_delete_documents(self, client):
        """Test deleting documents"""
        # Add documents
        ids = client.add_documents(["Doc 1", "Doc 2", "Doc 3"])

        # Delete one
        client.delete_documents([ids[0]])

        assert client.document_count == 2

    def test_reset(self, client):
        """Test resetting the store"""
        client.add_documents(["Doc 1", "Doc 2"])
        assert client.document_count == 2

        client.reset()
        assert client.document_count == 0

    def test_empty_query_raises_error(self, client):
        """Test that empty query raises ValueError"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            client.search("")

    def test_empty_document_raises_error(self, client):
        """Test that empty document raises ValueError"""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            client.add_document("")


class TestConfiguration:
    """Test configuration management"""

    def test_default_chinese_config(self):
        """Test default Chinese configuration"""
        config = RAGConfig.default_chinese()

        assert config.embedding.provider == "text2vec"
        assert "chinese" in config.embedding.model_name.lower()
        assert config.vector_store.provider == "chromadb"
        assert config.enable_reranking is True

    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = RAGConfig.default_chinese()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "embedding" in config_dict
        assert "vector_store" in config_dict

    def test_config_from_dict(self):
        """Test configuration deserialization"""
        config_dict = {
            "embedding": {"provider": "text2vec", "model_name": "test-model"},
            "vector_store": {"provider": "chromadb"},
            "enable_reranking": False,
        }

        config = RAGConfig.from_dict(config_dict)

        assert config.embedding.model_name == "test-model"
        assert config.enable_reranking is False


class TestReranking:
    """Test reranking functionality"""

    def test_reranking_enabled(self):
        """Test that reranking can be enabled"""
        embedding = MockEmbedding()
        store = MockVectorStore()

        client = RAGClient(
            embedding=embedding, vector_store=store, enable_reranking=True
        )

        assert client.enable_reranking is True
        assert client.reranker is not None

    def test_reranking_disabled(self):
        """Test that reranking can be disabled"""
        embedding = MockEmbedding()
        store = MockVectorStore()

        # Need to provide config with reranking disabled, since default enables it
        config = RAGConfig.default_chinese()
        config.enable_reranking = False

        client = RAGClient(
            embedding=embedding, vector_store=store, config=config, enable_reranking=False
        )

        assert client.enable_reranking is False
        assert client.reranker is None


class TestStatistics:
    """Test statistics and metadata"""

    def test_get_stats(self):
        """Test getting system statistics"""
        embedding = MockEmbedding(dimension=256)
        store = MockVectorStore()

        client = RAGClient(embedding=embedding, vector_store=store)

        stats = client.get_stats()

        assert "embedding_model" in stats
        assert "embedding_dimension" in stats
        assert stats["embedding_dimension"] == 256
        assert "vector_store" in stats
        assert "document_count" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
