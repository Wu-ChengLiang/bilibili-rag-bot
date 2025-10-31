"""TDD 测试：交互式 RAG 对话"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.client import RAGClient
from src.rag.llm_client import LLMClient


class TestRAGChatbot:
    """测试 RAGChatbot 对话功能"""

    @pytest.fixture
    def mock_rag_client(self):
        """创建 RAG 客户端"""
        client = RAGClient()
        # 添加测试数据
        client.add_documents([
            "向量数据库是存储和查询高维向量的数据库系统",
            "RAG（检索增强生成）结合向量搜索和 LLM 能力",
            "Embedding 是将文本转换为数值向量的技术",
        ])
        return client

    def test_search_retrieves_documents(self, mock_rag_client):
        """测试：搜索能检索到相关文档"""
        results = mock_rag_client.search("向量数据库", limit=1)

        assert len(results) > 0
        assert "向量数据库" in results[0]["content"]

    def test_chatbot_can_be_initialized(self):
        """测试：RAGChatbot 可以初始化"""
        # 这会失败，因为还没有实现
        from main import RAGChatbot

        chatbot = RAGChatbot(
            data_directory="./tests/fixtures",
            llm_api_key="test_key"
        )
        assert chatbot is not None

    @patch('main.LLMClient')
    def test_chatbot_can_answer_question(self, mock_llm_client):
        """测试：RAGChatbot 可以回答问题"""
        from main import RAGChatbot

        # Mock LLM 客户端
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "向量数据库是存储和查询高维向量的系统"
        mock_llm_client.return_value = mock_instance

        chatbot = RAGChatbot(
            data_directory="./tests/fixtures",
            llm_api_key="test_key"
        )

        response = chatbot.chat("什么是向量数据库？")
        assert response is not None
        assert isinstance(response, str)

    @patch('main.LLMClient')
    def test_chatbot_returns_context(self, mock_llm_client):
        """测试：RAGChatbot 返回检索的上下文"""
        from main import RAGChatbot

        # Mock LLM 客户端
        mock_instance = MagicMock()
        mock_instance.generate.return_value = "RAG 是检索增强生成的缩写"
        mock_llm_client.return_value = mock_instance

        chatbot = RAGChatbot(
            data_directory="./tests/fixtures",
            llm_api_key="test_key"
        )

        result = chatbot.chat_with_context("什么是 RAG？")

        # 验证返回格式
        assert "answer" in result
        assert "context" in result
        assert "query" in result

    def test_chatbot_supports_multiple_data_sources(self):
        """测试：RAGChatbot 支持多种数据源"""
        from main import RAGChatbot

        chatbot = RAGChatbot(
            # 支持本地文件和飞书
            local_directory="./docs",
            feishu_doc_ids=["doc_id_1", "doc_id_2"],
            llm_api_key="test_key"
        )

        assert chatbot is not None

    def test_chatbot_interactive_loop(self):
        """测试：RAGChatbot 支持交互循环"""
        from main import RAGChatbot

        chatbot = RAGChatbot(
            data_directory="./tests/fixtures",
            llm_api_key="test_key"
        )

        # 验证有交互方法
        assert hasattr(chatbot, "interactive_chat")
        assert callable(chatbot.interactive_chat)


class TestRAGChatbotIntegration:
    """集成测试"""

    @patch('main.LLMClient')
    def test_full_conversation_flow(self, mock_llm_client):
        """测试：完整的对话流程"""
        from main import RAGChatbot

        # Mock LLM 客户端
        mock_instance = MagicMock()
        mock_instance.generate.side_effect = [
            "RAG 是检索增强生成",
            "RAG 的优势包括更准确的回答",
        ]
        mock_llm_client.return_value = mock_instance

        chatbot = RAGChatbot(
            data_directory="./tests/fixtures",
            llm_api_key="test_key"
        )

        # 模拟对话
        response1 = chatbot.chat("什么是 RAG？")
        response2 = chatbot.chat("它有什么优势？")

        assert response1 is not None
        assert response2 is not None

    def test_chatbot_graceful_exit(self):
        """测试：优雅退出"""
        from main import RAGChatbot

        chatbot = RAGChatbot(
            data_directory="./tests/fixtures",
            llm_api_key="test_key"
        )

        # 应该能处理退出命令
        result = chatbot.should_exit("exit")
        assert result is True

        result = chatbot.should_exit("quit")
        assert result is True

        result = chatbot.should_exit("你好")
        assert result is False
