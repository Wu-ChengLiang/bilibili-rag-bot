"""RAG + 多轮对话服务 - 整合 RAG、LLM、历史管理"""

import logging
from typing import List, Optional, Dict, Any

from src.rag.client import RAGClient
from src.llm.factory import LLMFactory
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


class RAGChatService:
    """完整的 RAG + 多轮对话 + 历史管理服务"""

    def __init__(
        self,
        llm_provider: str,
        llm_api_key: str,
        data_directory: Optional[str] = None,
        llm_model: str = "moonshot-v1-8k",
        history_dir: str = "./history",
    ):
        """
        初始化服务

        Args:
            llm_provider: LLM 提供者 ("kimi", "gpt", etc.)
            llm_api_key: LLM API 密钥
            data_directory: 数据目录（用于 RAG）
            llm_model: LLM 模型名称
            history_dir: 历史文件存储目录
        """
        logger.info("初始化 RAGChatService...")

        # 初始化 RAG 客户端
        self.rag_client = RAGClient()
        if data_directory:
            self._load_documents(data_directory)

        # 初始化 LLM 客户端
        self.llm = LLMFactory.create(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
        )

        # 初始化历史管理
        self.conversation_mgr = ConversationManager(base_dir=history_dir)

        logger.info("✅ RAGChatService 初始化完成")

    def _load_documents(self, data_directory: str) -> None:
        """从目录加载文档"""
        from src.data.loaders import LocalFileLoader

        logger.info(f"从 {data_directory} 加载文档...")

        # 加载 .txt 文件
        loader_txt = LocalFileLoader(directory=data_directory, file_pattern="*.txt")
        documents_txt = loader_txt.load()

        # 加载 .md 文件
        loader_md = LocalFileLoader(directory=data_directory, file_pattern="*.md")
        documents_md = loader_md.load()

        documents = documents_txt + documents_md

        if documents:
            for doc in documents:
                self.rag_client.add_document(
                    content=doc.content,
                    metadata={
                        "source": doc.source,
                        "title": doc.title or "Untitled",
                        "doc_id": doc.doc_id,
                    },
                    doc_id=doc.doc_id,
                )
            logger.info(f"✅ 加载了 {len(documents)} 个文档")
        else:
            logger.warning("⚠️  未加载任何文档")

    def chat(
        self,
        platform: str,
        user_id: str,
        user_name: str,
        message: str,
        use_history: bool = True,
        search_limit: int = 3,
    ) -> str:
        """
        处理用户消息，返回回复

        Args:
            platform: 平台名称 (bilibili, weibo, etc.)
            user_id: 用户 ID
            user_name: 用户名
            message: 用户消息
            use_history: 是否使用对话历史
            search_limit: 搜索结果限制

        Returns:
            LLM 的回复
        """
        logger.info(f"[{platform}/{user_id}] 收到消息: {message[:50]}...")

        # 1. RAG 搜索相关文档
        try:
            # 确保文本编码正确
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            search_results = self.rag_client.search(message, limit=search_limit)
            context = [result["content"] for result in search_results]
        except Exception as e:
            logger.warning(f"搜索失败: {e}")
            context = []

        if not context:
            logger.warning(f"未找到相关文档")
            context = ["（未找到相关文档）"]

        # 2. 加载对话历史
        history = []
        if use_history:
            history = self.conversation_mgr.get_latest_messages(
                platform, user_id, limit=5
            )
            logger.info(f"加载 {len(history)} 条历史记录")

        # 3. 调用 LLM 生成回答
        try:
            if history:
                # 有历史 - 使用多轮对话
                reply = self.llm.generate_with_history(
                    query=message,
                    context=context,
                    history=history,
                )
            else:
                # 无历史 - 单轮对话
                reply = self.llm.generate(
                    query=message,
                    context=context,
                )

            logger.info(f"✅ 生成回答: {reply[:50]}...")
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = f"抱歉，生成回答时出现错误: {str(e)}"

        # 4. 保存到历史
        try:
            self.conversation_mgr.add_message(
                platform=platform,
                user_id=user_id,
                role="user",
                content=message,
            )
            self.conversation_mgr.add_message(
                platform=platform,
                user_id=user_id,
                role="assistant",
                content=reply,
            )
            logger.info(f"✅ 历史已保存")
        except Exception as e:
            logger.error(f"保存历史失败: {e}")

        return reply

    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        rag_stats = self.rag_client.get_stats()
        return {
            **rag_stats,
            "llm_provider": self.llm.__class__.__name__,
            "service": "RAGChatService",
        }

    def clear_user_history(self, platform: str, user_id: str) -> None:
        """清空用户的对话历史"""
        self.conversation_mgr.clear_history(platform, user_id)
        logger.info(f"已清空 {platform}/{user_id} 的对话历史")
