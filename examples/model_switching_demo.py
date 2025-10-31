"""Demo: Elegant model switching with RAGClient

This script demonstrates how to easily switch between different
embedding models using the configuration system.
"""

from src.rag import RAGClient
from src.rag.config import RAGConfig, EmbeddingConfig

# Sample documents for testing
SAMPLE_DOCS = [
    "Python 是一门高级编程语言，广泛应用于数据科学、机器学习和网络开发。",
    "RAG 系统结合了检索和生成技术，能够提供更准确和有根据的答案。",
    "向量数据库用于存储和检索高维向量，是现代 AI 系统的核心组件。",
    "中文文本处理需要特殊的分词和嵌入模型来理解语言的语义。",
]

QUERIES = [
    "什么是RAG系统？",
    "Python有什么用途？",
    "向量数据库的作用",
]


def demo_model_switching():
    """Demo switching between different embedding models"""

    print("=" * 60)
    print("RAG 模型切换演示")
    print("=" * 60)

    # Model configurations to test
    models = [
        {
            "provider": "text2vec",
            "model_name": "shibing624/text2vec-base-chinese",
            "description": "Text2Vec-base (快速，轻量)"
        },
        {
            "provider": "gte",
            "model_name": "thenlper/gte-base-zh",
            "description": "GTE-base (快速，精度好)"
        },
    ]

    for model_config in models:
        print(f"\n{'─' * 60}")
        print(f"📊 测试模型: {model_config['description']}")
        print(f"   Provider: {model_config['provider']}")
        print(f"   Model: {model_config['model_name']}")
        print(f"{'─' * 60}")

        # Create RAG client with specific embedding model
        config = RAGConfig(
            embedding=EmbeddingConfig(
                provider=model_config["provider"],
                model_name=model_config["model_name"]
            )
        )

        client = RAGClient.from_config(config)

        # Print system stats
        stats = client.get_stats()
        print(f"\n📈 系统信息:")
        print(f"   嵌入模型: {stats['embedding_model']}")
        print(f"   向量维度: {stats['embedding_dimension']}")

        # Add documents
        print(f"\n➕ 添加 {len(SAMPLE_DOCS)} 个文档...")
        for i, doc in enumerate(SAMPLE_DOCS, 1):
            client.add_document(doc, metadata={"index": i})

        print(f"✓ 已添加 {client.document_count} 个文档")

        # Test search with different queries
        print(f"\n🔍 搜索测试:")
        for query in QUERIES:
            results = client.search(query, limit=2)
            print(f"\n   查询: \"{query}\"")
            for i, result in enumerate(results, 1):
                score = result.similarity_score
                print(f"   [{i}] 相似度: {score:.4f}")
                print(f"       内容: {result.content[:50]}...")

        # Clean up
        client.reset()
        print(f"\n✓ 已清空数据库\n")


def demo_simple_usage():
    """Show the simplest way to use different models"""

    print("\n" + "=" * 60)
    print("快速开始：如何切换模型")
    print("=" * 60)

    print("\n1️⃣  使用默认模型 (text2vec):")
    print("""
    from src.rag import RAGClient

    client = RAGClient()  # 使用默认的 text2vec
    """)

    print("\n2️⃣  切换到 GTE 模型:")
    print("""
    from src.rag import RAGClient
    from src.rag.config import RAGConfig, EmbeddingConfig

    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="gte",
            model_name="thenlper/gte-base-zh"
        )
    )
    client = RAGClient.from_config(config)
    """)

    print("\n3️⃣  直接注入 embedding 对象:")
    print("""
    from src.rag import RAGClient
    from src.rag.embeddings import GTEEmbedding

    embedding = GTEEmbedding(model_name="thenlper/gte-large-zh")
    client = RAGClient(embedding=embedding)
    """)

    print("\n4️⃣  使用工厂方法:")
    print("""
    from src.rag import RAGClient
    from src.rag.embeddings import create_embedding

    embedding = create_embedding(
        provider="gte",
        model_name="thenlper/gte-base-zh"
    )
    client = RAGClient(embedding=embedding)
    """)


if __name__ == "__main__":
    # Run demo (disabled by default - remove models to test locally)
    print("\n💡 提示: 下面的演示需要下载模型，这会花费一些时间。")
    print("   要运行完整演示，请取消注释 demo_model_switching() 这一行。\n")

    # demo_model_switching()  # Uncomment to run full demo

    # Show usage examples
    demo_simple_usage()

    print("\n" + "=" * 60)
    print("✓ 演示完成！")
    print("=" * 60)
