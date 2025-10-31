"""
RAG 搜索调试工具
用于分析 RAG 搜索过程和相似度评分
"""

import os
from dotenv import load_dotenv
from src.services.rag_chat_service import RAGChatService

load_dotenv()

def main():
    # 初始化服务
    service = RAGChatService(
        llm_provider="kimi",
        llm_api_key=os.getenv("MOONSHOT_API_KEY"),
        data_directory="./data"
    )

    print("\n" + "=" * 80)
    print("RAG 搜索调试工具")
    print("=" * 80)
    print("输入查询词，查看搜索过程和相似度评分")
    print("输入 'exit' 或 'quit' 退出\n")

    while True:
        try:
            query = input("搜索: ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 再见！")
                break

            # 调用 debug_search
            service.debug_search(query, limit=5)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
