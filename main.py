"""
RAG 交互式多轮对话系统 - main.py
支持从本地文件加载数据，使用 RAG + LLM 进行多轮对话，自动维护对话历史
"""

import logging
import os
from dotenv import load_dotenv

from src.services.rag_chat_service import RAGChatService

# 加载 .env 文件
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 启动交互式对话"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG 交互式多轮对话系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python3 main.py                              # 使用默认配置
  python3 main.py --data-dir ./docs            # 指定数据目录
  python3 main.py --llm-api-key sk-xxx         # 指定 API 密钥
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        help="数据目录（默认从环境变量或 ./data）",
        default=os.getenv("DATA_DIRECTORY", "./data"),
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        help="LLM API 密钥（从环境变量或命令行）",
        default=os.getenv("MOONSHOT_API_KEY"),
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        help="LLM 提供者",
        default="kimi",
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        help="对话历史存储目录",
        default="./history",
    )

    args = parser.parse_args()

    # 验证必要参数
    if not args.llm_api_key:
        logger.error("❌ 未提供 LLM API 密钥")
        logger.info("设置方式：")
        logger.info("  1. 命令行: python main.py --llm-api-key sk-xxx")
        logger.info("  2. 环境变量: export MOONSHOT_API_KEY=sk-xxx")
        return

    # 初始化服务
    try:
        logger.info("初始化 RAG 服务...")
        service = RAGChatService(
            llm_provider=args.llm_provider,
            llm_api_key=args.llm_api_key,
            data_directory=args.data_dir,
            history_dir=args.history_dir,
        )
        logger.info("✅ 服务初始化成功")
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return

    # 显示欢迎信息
    print("\n" + "=" * 60)
    print("🤖 RAG 智能问答系统（支持多轮对话）")
    print("=" * 60)
    print("输入你的问题，系统将根据文档回答")
    print("系统会自动记忆对话历史")
    print("输入 'exit', 'quit' 或 'q' 退出\n")

    # 交互式对话循环
    platform = "terminal"
    user_id = "default"
    user_name = "用户"

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            # 退出检查
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 再见！")
                break

            # 处理对话
            print("\n🔍 思考中...\n")
            reply = service.chat(
                platform=platform,
                user_id=user_id,
                user_name=user_name,
                message=user_input,
            )

            print(f"助手: {reply}\n")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            logger.error(f"处理失败: {e}")
            print(f"❌ 发生错误: {e}\n")


if __name__ == "__main__":
    main()
