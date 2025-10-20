"""完整的 RAG 演示：文档加载 → 分块 → 检索 → Rerank → LLM 生成"""

import os
from src.rag.client import RAGClient
from src.rag.document_loader import DocumentLoader
from src.rag.llm_client import LLMClient
from src.rag.reranker import Reranker


def main():
    print("=" * 80)
    print("完整 RAG 系统演示")
    print("=" * 80)

    # API Key
    KIMI_API_KEY = os.getenv("MOONSHOT_API_KEY", "sk-QMVGFIxphgo70al8We9W76woZIhz2dER0VyfZb0DSRwHPrlO")

    # 步骤 1: 加载文档并分块
    print("\n[步骤 1] 加载 data 目录下所有文档并分块...")
    loader = DocumentLoader()

    # 加载所有文档
    data_files = [
        "data/fire2.txt",
        "data/life3.txt",
        "data/life4.txt"
    ]

    all_chunks = []
    for file_path in data_files:
        file_chunks = loader.load_file(
            file_path,
            strategy="smart",
            chunk_size=300
        )
        all_chunks.extend(file_chunks)
        print(f"   - {file_path}: {len(file_chunks)} 个块")

    chunks = all_chunks
    print(f"   总计加载了 {len(chunks)} 个文档块")
    print(f"   示例块: {chunks[0][:100]}..." if chunks else "   无数据")

    # 步骤 2: 初始化 RAG 客户端
    print("\n[步骤 2] 初始化 RAG 客户端...")
    rag_client = RAGClient(
        persist_directory="./full_demo_db",
        collection_name="life_rag"
    )

    # 步骤 3: 添加文档到向量数据库
    print("\n[步骤 3] 添加文档到向量数据库...")
    doc_ids = rag_client.add_documents(chunks)
    print(f"   已添加 {len(doc_ids)} 个文档块")

    # 步骤 4: 用户提问
    query = "我加入的初创公司叫什么名字"
    print(f"\n[步骤 4] 用户提问")
    print(f"   问题: {query}")

    # 步骤 5: 向量检索
    print("\n[步骤 5] 向量检索...")
    search_results = rag_client.search(query, limit=10)
    print(f"   检索到 {len(search_results)} 个相关文档")

    # 步骤 6: Rerank 重排序
    print("\n[步骤 6] Rerank 重排序...")
    reranker = Reranker()
    reranked_results = reranker.rerank(query, search_results, top_k=3)
    print(f"   重排后保留 Top-{len(reranked_results)} 文档")

    print("\n   Top 3 文档片段:")
    for i, result in enumerate(reranked_results, 1):
        print(f"\n   {i}. [相似度: {result['score']:.3f}] [Rerank分数: {result.get('rerank_score', 0):.3f}]")
        print(f"      {result['content'][:150]}...")

    # 步骤 7: LLM 生成回答
    print("\n[步骤 7] LLM 生成回答...")
    llm_client = LLMClient(api_key=KIMI_API_KEY)

    context_docs = [r["content"] for r in reranked_results]

    try:
        answer = llm_client.generate(
            query=query,
            context=context_docs
        )

        print("\n" + "=" * 80)
        print("最终回答")
        print("=" * 80)
        print(answer)
        print("=" * 80)

    except Exception as e:
        print(f"\n   ⚠️  LLM 调用失败: {e}")
        print("   但检索和 Rerank 功能正常工作！")

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
