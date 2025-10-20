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
    print("\n[步骤 1] 加载文档并分块...")
    loader = DocumentLoader()
    chunks = loader.load_file(
        "data/life3.txt",
        strategy="smart",
        chunk_size=300
    )
    print(f"   加载了 {len(chunks)} 个文档块")
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
    query = "福贵是谁？他和老牛有什么故事？"
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

    # 演示多轮对话
    print("\n" + "=" * 80)
    print("多轮对话演示")
    print("=" * 80)

    questions = [
        "老牛叫什么名字？",
        "福贵的家里有多少亩地？",
        "文中提到的城市有哪些？"
    ]

    for q in questions:
        print(f"\n问: {q}")

        # 检索
        results = rag_client.search(q, limit=5)

        # Rerank
        reranked = reranker.rerank(q, results, top_k=2)

        print(f"检索到 {len(results)} 个文档，Rerank 后保留 {len(reranked)} 个")

        # 显示最相关的文档
        if reranked:
            print(f"最相关文档: {reranked[0]['content'][:100]}...")

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)

    print("\n功能总结:")
    print("✅ 文档分块 (DocumentLoader)")
    print("✅ 向量检索 (ChromaDB + Text2Vec)")
    print("✅ Rerank 重排序 (基于关键词和长度)")
    print("✅ LLM 生成 (Kimi API)")


if __name__ == "__main__":
    main()
