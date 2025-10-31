"""
RAG 交互式对话系统 - main.py
支持从本地文件和飞书加载数据，使用 RAG + LLM 进行对话
"""

import logging
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.rag.client import RAGClient
from src.rag.llm_client import LLMClient
from src.data.loaders import LocalFileLoader, FeishuDocxLoader
from src.data.config import FeishuConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """RAG 对话机器人 - 支持多数据源"""

    def __init__(
        self,
        llm_api_key: str,
        llm_model: str = "moonshot-v1-8k",
        local_directory: Optional[str] = None,
        data_directory: Optional[str] = None,
        feishu_doc_ids: Optional[List[str]] = None,
        feishu_config: Optional[FeishuConfig] = None,
    ):
        """
        初始化 RAGChatbot

        Args:
            llm_api_key: LLM API 密钥（Kimi API）
            llm_model: LLM 模型名称
            local_directory: 本地文件目录
            data_directory: 数据目录（会扫描所有 .txt 和 .md 文件）
            feishu_doc_ids: 飞书文档 ID 列表
            feishu_config: 飞书配置对象
        """
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model

        # 初始化 RAG 客户端
        logger.info("初始化 RAG 客户端...")
        self.rag_client = RAGClient()

        # 加载数据源
        documents = []

        # 本地目录
        if data_directory:
            logger.info(f"从目录加载文件: {data_directory}")
            loader = LocalFileLoader(directory=data_directory, file_pattern="*.{txt,md}")
            documents.extend(loader.load())

        # 特定文件目录
        if local_directory:
            logger.info(f"从目录加载文件: {local_directory}")
            loader = LocalFileLoader(directory=local_directory, file_pattern="*.{txt,md}")
            documents.extend(loader.load())

        # 飞书文档
        if feishu_doc_ids:
            try:
                if feishu_config is None:
                    feishu_config = FeishuConfig.from_env()

                logger.info(f"从飞书加载 {len(feishu_doc_ids)} 个文档...")
                loader = FeishuDocxLoader(config=feishu_config, document_ids=feishu_doc_ids)
                documents.extend(loader.load())
            except Exception as e:
                logger.warning(f"飞书加载失败: {e}")

        # 添加文档到 RAG
        if documents:
            logger.info(f"添加 {len(documents)} 个文档到向量存储...")
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
            logger.info("✅ 文档加载完成")
        else:
            logger.warning("⚠️  未加载任何文档")

        # 初始化 LLM 客户端
        if llm_api_key:
            logger.info("初始化 LLM 客户端...")
            self.llm_client = LLMClient(api_key=llm_api_key, model=llm_model)
        else:
            logger.warning("未提供 LLM API 密钥，仅搜索功能可用")
            self.llm_client = None

        logger.info("✅ RAGChatbot 初始化完成")

    def chat(self, query: str, limit: int = 3) -> str:
        """
        简单对话 - 搜索 + LLM 生成

        Args:
            query: 用户问题
            limit: 检索结果数量

        Returns:
            LLM 生成的回答
        """
        if not self.llm_client:
            return "❌ LLM 未初始化，请提供 API 密钥"

        # 搜索相关文档
        results = self.rag_client.search(query, limit=limit)

        if not results:
            return "未找到相关文档"

        # 提取内容
        context = [result["content"] for result in results]

        # 使用 LLM 生成回答
        response = self.llm_client.generate(query=query, context=context)
        return response

    def chat_with_context(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        对话并返回上下文

        Args:
            query: 用户问题
            limit: 检索结果数量

        Returns:
            包含 answer, context, query 的字典
        """
        if not self.llm_client:
            return {
                "answer": "❌ LLM 未初始化",
                "context": [],
                "query": query,
            }

        # 搜索相关文档
        results = self.rag_client.search(query, limit=limit)

        # 提取内容和元数据
        context = []
        if results:
            for result in results:
                context.append({
                    "content": result["content"],
                    "score": result["score"],
                    "metadata": result.get("metadata", {}),
                })

        # 生成回答
        context_text = [r["content"] for r in context]
        answer = self.llm_client.generate(query=query, context=context_text) if context_text else "未找到相关文档"

        return {
            "query": query,
            "answer": answer,
            "context": context,
        }

    def search_only(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        仅搜索，不使用 LLM

        Args:
            query: 查询
            limit: 结果数量

        Returns:
            搜索结果列表
        """
        return self.rag_client.search(query, limit=limit)

    def should_exit(self, user_input: str) -> bool:
        """判断是否应该退出"""
        exit_commands = ["exit", "quit", "q", "bye", "goodbye", "退出", "再见"]
        return user_input.lower().strip() in exit_commands

    def interactive_chat(self) -> None:
        """
        交互式对话循环
        在终端运行持续对话
        """
        print("\n" + "=" * 60)
        print("🤖 RAG 智能问答系统")
        print("=" * 60)
        print("输入你的问题，系统将根据文档回答")
        print("输入 'exit' 或 'quit' 退出\n")

        while True:
            try:
                user_input = input("你: ").strip()

                if not user_input:
                    continue

                if self.should_exit(user_input):
                    print("\n👋 再见！")
                    break

                # 对话
                print("\n🔍 搜索中...\n")
                response = self.chat(user_input)

                print(f"助手: {response}\n")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                logger.error(f"错误: {e}")
                print(f"❌ 发生错误: {e}\n")

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return self.rag_client.get_stats()


def main():
    """主函数 - 启动交互式对话"""
    import argparse
    from dotenv import load_dotenv

    # 加载 .env 文件
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG 交互式对话系统")
    parser.add_argument(
        "--local-dir",
        type=str,
        help="本地文件目录",
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="数据目录（会扫描所有 .txt 和 .md 文件）",
        default=os.getenv("DATA_DIRECTORY", "./docs"),
    )
    parser.add_argument(
        "--feishu-doc-ids",
        type=str,
        nargs="+",
        help="飞书文档 ID",
        default=None,
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        help="LLM API 密钥",
        default=None,
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM 模型名称",
        default="moonshot-v1-8k",
    )

    args = parser.parse_args()

    # 获取 API 密钥
    llm_api_key = args.llm_api_key or os.getenv("MOONSHOT_API_KEY")

    if not llm_api_key:
        logger.warning("⚠️  未提供 LLM API 密钥，仅搜索功能可用")
        logger.info("设置方式：")
        logger.info("  1. 命令行: python main.py --llm-api-key xxx")
        logger.info("  2. 环境变量: export MOONSHOT_API_KEY=xxx")

    # 创建对话机器人
    chatbot = RAGChatbot(
        llm_api_key=llm_api_key or "dummy_key",
        llm_model=args.llm_model,
        local_directory=args.local_dir,
        data_directory=args.data_dir,
        feishu_doc_ids=args.feishu_doc_ids,
    )

    # 显示统计信息
    stats = chatbot.get_stats()
    print(f"\n📊 系统统计: {stats}\n")

    # 启动交互式对话
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()
