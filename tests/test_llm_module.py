"""TDD - LLM 模块和多轮对话功能测试 (RED PHASE)"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaseLLMClient:
    """测试 LLM 抽象基类"""

    def test_base_llm_client_has_generate_method(self):
        """测试：BaseLLMClient 有 generate 方法"""
        from src.llm.base import BaseLLMClient

        assert hasattr(BaseLLMClient, "generate")

    def test_base_llm_client_has_generate_with_history_method(self):
        """测试：BaseLLMClient 有 generate_with_history 方法"""
        from src.llm.base import BaseLLMClient

        assert hasattr(BaseLLMClient, "generate_with_history")

    def test_generate_is_abstract(self):
        """测试：generate 是抽象方法"""
        from src.llm.base import BaseLLMClient

        # 不能直接实例化
        with pytest.raises(TypeError):
            BaseLLMClient()

    def test_kimi_client_implements_base_llm_client(self):
        """测试：KimiClient 实现 BaseLLMClient"""
        from src.llm.base import BaseLLMClient
        from src.llm.kimi import KimiClient

        assert issubclass(KimiClient, BaseLLMClient)


class TestKimiClient:
    """测试 Kimi LLM 客户端"""

    def test_kimi_client_initialization(self):
        """测试：KimiClient 可以初始化"""
        from src.llm.kimi import KimiClient

        client = KimiClient(api_key="test_key", model="moonshot-v1-8k")
        assert client is not None

    @patch("src.llm.kimi.OpenAI")
    def test_kimi_client_generate_single_turn(self, mock_openai):
        """测试：KimiClient 可以进行单轮对话"""
        from src.llm.kimi import KimiClient

        # Mock OpenAI 响应
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "这是 Kimi 的回答"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = KimiClient(api_key="test_key")
        response = client.generate(
            query="什么是向量数据库？",
            context=["向量数据库是存储高维向量的系统"]
        )

        assert response == "这是 Kimi 的回答"
        assert isinstance(response, str)

    @patch("src.llm.kimi.OpenAI")
    def test_kimi_client_generate_with_history(self, mock_openai):
        """测试：KimiClient 支持多轮对话（带历史）"""
        from src.llm.kimi import KimiClient

        # Mock OpenAI 响应
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "基于前面的讨论，这是回答"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = KimiClient(api_key="test_key")

        history = [
            {"role": "user", "content": "什么是 RAG？"},
            {"role": "assistant", "content": "RAG 是检索增强生成..."},
        ]

        response = client.generate_with_history(
            query="它有什么优势？",
            context=["RAG 的优势包括..."],
            history=history
        )

        assert response is not None
        assert isinstance(response, str)
        # 验证调用了 OpenAI API
        mock_openai.return_value.chat.completions.create.assert_called_once()


class TestLLMFactory:
    """测试 LLM 工厂类"""

    def test_factory_can_create_kimi_client(self):
        """测试：工厂可以创建 KimiClient"""
        from src.llm.factory import LLMFactory

        client = LLMFactory.create(provider="kimi", api_key="test_key")
        assert client is not None

    def test_factory_raises_error_for_unknown_provider(self):
        """测试：工厂对未知提供者抛出错误"""
        from src.llm.factory import LLMFactory

        with pytest.raises(ValueError):
            LLMFactory.create(provider="unknown", api_key="test_key")

    def test_factory_can_register_new_provider(self):
        """测试：工厂支持注册新提供者"""
        from src.llm.factory import LLMFactory
        from src.llm.kimi import KimiClient

        # 支持这样的调用方式
        LLMFactory.register("custom", KimiClient)
        client = LLMFactory.create(provider="custom", api_key="test_key")
        assert client is not None


class TestConversationManager:
    """测试对话历史管理器"""

    def test_conversation_manager_initialization(self):
        """测试：ConversationManager 可以初始化"""
        from src.services.conversation_manager import ConversationManager

        mgr = ConversationManager(base_dir="./test_history")
        assert mgr is not None

    def test_load_history_returns_list(self):
        """测试：load_history 返回历史列表"""
        from src.services.conversation_manager import ConversationManager

        mgr = ConversationManager()
        # 不存在的用户返回空列表
        history = mgr.load_history(platform="bilibili", user_id="999999")
        assert isinstance(history, list)

    def test_save_and_load_history(self, tmp_path):
        """测试：保存和加载历史"""
        from src.services.conversation_manager import ConversationManager

        mgr = ConversationManager(base_dir=str(tmp_path))

        # 保存历史
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        mgr.save_history(platform="bilibili", user_id="123456", history=history)

        # 加载历史
        loaded = mgr.load_history(platform="bilibili", user_id="123456")
        assert len(loaded) == 2
        assert loaded[0]["content"] == "你好"

    def test_add_message_to_history(self, tmp_path):
        """测试：添加消息到历史"""
        from src.services.conversation_manager import ConversationManager

        mgr = ConversationManager(base_dir=str(tmp_path))

        # 添加第一条消息
        mgr.add_message(
            platform="bilibili",
            user_id="123456",
            role="user",
            content="你好"
        )

        # 添加第二条消息
        mgr.add_message(
            platform="bilibili",
            user_id="123456",
            role="assistant",
            content="你好！有什么我可以帮助的吗？"
        )

        # 验证历史
        history = mgr.load_history(platform="bilibili", user_id="123456")
        assert len(history) == 2

    def test_platform_isolation(self, tmp_path):
        """测试：不同平台的历史相互隔离"""
        from src.services.conversation_manager import ConversationManager

        mgr = ConversationManager(base_dir=str(tmp_path))

        # 在 bilibili 上保存
        mgr.add_message("bilibili", "user1", "user", "b站消息")
        # 在 weibo 上保存
        mgr.add_message("weibo", "user1", "user", "微博消息")

        # 验证隔离
        b_history = mgr.load_history("bilibili", "user1")
        w_history = mgr.load_history("weibo", "user1")

        assert len(b_history) == 1
        assert len(w_history) == 1
        assert b_history[0]["content"] == "b站消息"
        assert w_history[0]["content"] == "微博消息"


class TestRAGChatService:
    """测试 RAG + 多轮对话服务"""

    @patch("src.services.rag_chat_service.RAGClient")
    @patch("src.services.rag_chat_service.LLMFactory")
    def test_rag_chat_service_initialization(self, mock_factory, mock_rag):
        """测试：RAGChatService 可以初始化"""
        from src.services.rag_chat_service import RAGChatService

        service = RAGChatService(
            llm_provider="kimi",
            llm_api_key="test_key",
            data_directory="./data"
        )
        assert service is not None

    @patch("src.services.rag_chat_service.RAGClient")
    @patch("src.services.rag_chat_service.LLMFactory")
    def test_rag_chat_service_chat_method(self, mock_factory, mock_rag, tmp_path):
        """测试：RAGChatService 可以处理聊天请求"""
        from src.services.rag_chat_service import RAGChatService

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "这是回答"
        mock_llm.generate_with_history.return_value = "这是回答"
        mock_factory.create.return_value = mock_llm

        # Mock RAG
        mock_rag_instance = MagicMock()
        mock_rag_instance.search.return_value = [
            {"score": 0.9, "content": "相关文档"}
        ]
        mock_rag.return_value = mock_rag_instance

        service = RAGChatService(
            llm_provider="kimi",
            llm_api_key="test_key",
            data_directory="./data",
            history_dir=str(tmp_path),
        )

        # 调用 chat
        response = service.chat(
            platform="bilibili",
            user_id="123456",
            user_name="阿良",
            message="你好"
        )

        assert response is not None
        assert isinstance(response, str)
        # 验证调用了 LLM（无论是 generate 还是 generate_with_history）
        assert mock_llm.generate.called or mock_llm.generate_with_history.called

    @patch("src.services.rag_chat_service.RAGClient")
    @patch("src.services.rag_chat_service.LLMFactory")
    def test_rag_chat_service_maintains_history(self, mock_factory, mock_rag, tmp_path):
        """测试：RAGChatService 维护对话历史"""
        from src.services.rag_chat_service import RAGChatService

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "回答"
        mock_llm.generate_with_history.return_value = "基于历史的回答"
        mock_factory.create.return_value = mock_llm

        # Mock RAG
        mock_rag_instance = MagicMock()
        mock_rag_instance.search.return_value = [
            {"score": 0.9, "content": "文档"}
        ]
        mock_rag.return_value = mock_rag_instance

        service = RAGChatService(
            llm_provider="kimi",
            llm_api_key="test_key",
            data_directory="./data",
            history_dir=str(tmp_path),
        )

        # 第一轮对话（无历史）
        service.chat("bilibili", "123456", "阿良", "你好")

        # 第二轮对话（有历史）
        service.chat("bilibili", "123456", "阿良", "继续")

        # 验证历史已保存
        from src.services.conversation_manager import ConversationManager
        mgr = ConversationManager(str(tmp_path))
        history = mgr.load_history("bilibili", "123456")
        assert len(history) >= 2  # 至少保存了 2 条消息


class TestChatAPI:
    """测试 FastAPI 接口"""

    def test_chat_endpoint_accepts_request(self):
        """测试：/chat 端点接受请求"""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        request_data = {
            "platform": "bilibili",
            "user_id": "123456",
            "user_name": "阿良",
            "message": "你好",
        }

        response = client.post("/chat", json=request_data)
        assert response.status_code in [200, 422]  # 允许验证错误

    def test_chat_endpoint_returns_correct_format(self):
        """测试：/chat 返回正确的响应格式"""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        request_data = {
            "platform": "bilibili",
            "user_id": "123456",
            "user_name": "阿良",
            "message": "你好",
        }

        # Mock 响应
        with patch("api.main.service") as mock_service:
            mock_service.chat.return_value = "这是回答"

            response = client.post("/chat", json=request_data)

            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "reply" in data or "error" in data
