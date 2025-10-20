"""Tests for RAG Client"""

import pytest
import os
import shutil
from src.rag.client import RAGClient


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path"""
    db_path = tmp_path / "test_chroma_db"
    yield str(db_path)
    # Cleanup
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


@pytest.fixture
def client(test_db_path):
    """Create a test client"""
    return RAGClient(persist_directory=test_db_path, collection_name="test_collection")


class TestRAGClientInitialization:
    """Test client initialization"""

    def test_client_creation(self, test_db_path):
        """Test that client can be created"""
        client = RAGClient(persist_directory=test_db_path)
        assert client is not None
        assert client.persist_directory == test_db_path

    def test_client_with_custom_collection(self, test_db_path):
        """Test client with custom collection name"""
        client = RAGClient(
            persist_directory=test_db_path,
            collection_name="custom_collection"
        )
        assert client.collection_name == "custom_collection"

    def test_persistence_directory_created(self, test_db_path):
        """Test that persistence directory is created"""
        RAGClient(persist_directory=test_db_path)
        assert os.path.exists(test_db_path)


class TestRAGClientDocumentOperations:
    """Test document operations"""

    def test_add_single_document(self, client):
        """Test adding a single document"""
        doc_id = client.add_document("这是一个测试文档")
        assert doc_id is not None
        assert isinstance(doc_id, str)

    def test_add_multiple_documents(self, client):
        """Test adding multiple documents"""
        documents = ["文档1", "文档2", "文档3"]
        doc_ids = client.add_documents(documents)
        assert len(doc_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)

    def test_add_document_with_metadata(self, client):
        """Test adding document with metadata"""
        doc_id = client.add_document(
            "文档内容",
            metadata={"source": "test", "type": "example"}
        )
        assert doc_id is not None

    def test_add_empty_document_raises_error(self, client):
        """Test that adding empty document raises error"""
        with pytest.raises(ValueError):
            client.add_document("")


class TestRAGClientSearch:
    """Test search functionality"""

    def test_search_returns_results(self, client):
        """Test that search returns results"""
        # Add some documents first
        client.add_documents(["中国的首都是北京", "法国的首都是巴黎", "日本的首都是东京"])

        # Search
        results = client.search("北京是哪个国家的首都", limit=1)
        assert len(results) > 0
        assert results[0]["content"] == "中国的首都是北京"

    def test_search_limit(self, client):
        """Test search result limit"""
        # Add documents
        for i in range(10):
            client.add_document(f"文档{i}")

        # Search with limit
        results = client.search("文档", limit=5)
        assert len(results) <= 5

    def test_search_empty_query_raises_error(self, client):
        """Test that empty query raises error"""
        with pytest.raises(ValueError):
            client.search("")

    def test_search_returns_scores(self, client):
        """Test that search results include scores"""
        client.add_documents(["测试文档1", "测试文档2"])
        results = client.search("测试", limit=2)

        assert all("score" in result for result in results)
        assert all(isinstance(result["score"], float) for result in results)


class TestRAGClientPersistence:
    """Test persistence functionality"""

    def test_data_persists_after_restart(self, test_db_path):
        """Test that data persists after client restart"""
        # Create client and add document
        client1 = RAGClient(persist_directory=test_db_path)
        client1.add_document("持久化测试文档")

        # Create new client instance
        client2 = RAGClient(persist_directory=test_db_path)

        # Search should find the document
        results = client2.search("持久化", limit=1)
        assert len(results) > 0
        assert "持久化" in results[0]["content"]
