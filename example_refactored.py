"""示例：展示重构后RAG系统的新特性"""

from src.rag import (
    RAGClient,
    RAGConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    DocumentLoader,
)


def example_1_simple_usage():
    """示例 1：最简单的用法（向后兼容）"""
    print("=" * 60)
    print("示例 1：简单用法")
    print("=" * 60)

    # 使用默认配置
    client = RAGClient()

    # 添加文档
    docs = ["这是第一个文档", "这是第二个文档", "这是第三个文档"]
    client.add_documents(docs)

    # 搜索
    results = client.search("文档", limit=2)

    print(f"找到 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content'][:30]}... (score: {result['score']:.3f})")

    # 获取统计信息
    stats = client.get_stats()
    print(f"\n系统统计: {stats['document_count']} 个文档")

    # 清理
    client.reset()
    print("\n✅ 示例 1 完成\n")


def example_2_config_based():
    """示例 2：使用配置对象"""
    print("=" * 60)
    print("示例 2：配置驱动")
    print("=" * 60)

    # 创建自定义配置
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="text2vec",
            model_name="shibing624/text2vec-base-chinese",
        ),
        vector_store=VectorStoreConfig(
            provider="chromadb",
            persist_directory="./demo_db",
            collection_name="my_docs",
        ),
        enable_reranking=True,
        default_search_limit=10,
    )

    # 从配置创建客户端
    client = RAGClient.from_config(config)

    # 或使用预设配置
    # client = RAGClient.default_chinese()

    # 使用文档加载器
    loader = DocumentLoader()
    # chunks = loader.load_file("your_file.txt", strategy="smart", chunk_size=300)

    # 模拟一些文档
    chunks = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的核心技术。",
        "深度学习使用神经网络进行学习。",
        "自然语言处理处理人类语言。",
        "RAG 系统结合检索和生成。",
    ]

    client.add_documents(chunks)

    # 搜索并重排序
    results = client.search("人工智能", limit=3)

    print(f"Reranking: {client.enable_reranking}")
    print(f"找到 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content']}")
        print(f"   Score: {result['score']:.3f}")

    # 清理
    client.reset()
    print("\n✅ 示例 2 完成\n")


def example_3_dependency_injection():
    """示例 3：依赖注入（高级用法）"""
    print("=" * 60)
    print("示例 3：依赖注入")
    print("=" * 60)

    from src.rag.embeddings import Text2VecEmbedding
    from src.rag.stores import ChromaDBStore

    # 手动创建组件
    embedding = Text2VecEmbedding(model_name="shibing624/text2vec-base-chinese")
    store = ChromaDBStore(
        persist_directory="./custom_db", collection_name="advanced_docs"
    )

    # 注入依赖
    client = RAGClient(embedding=embedding, vector_store=store, enable_reranking=True)

    # 使用客户端
    client.add_document("依赖注入使代码更易测试", metadata={"source": "best_practice"})

    results = client.search("测试")

    print(f"Embedding: {client.embedding.model_name}")
    print(f"Vector Store: {client.vector_store.__class__.__name__}")
    print(f"Results: {len(results)}")

    # 清理
    client.reset()
    print("\n✅ 示例 3 完成\n")


def example_4_testing_with_mocks():
    """示例 4：使用 Mock 进行测试"""
    print("=" * 60)
    print("示例 4：Mock 测试（演示可测试性）")
    print("=" * 60)

    from unittest.mock import Mock
    from src.rag import BaseEmbedding, BaseVectorStore

    # 创建 Mock 对象
    mock_embedding = Mock(spec=BaseEmbedding)
    mock_embedding.encode.return_value = [0.1] * 384
    mock_embedding.model_name = "MockEmbedding"
    mock_embedding.dimension = 384

    mock_store = Mock(spec=BaseVectorStore)
    mock_store.add_documents.return_value = ["doc_1", "doc_2"]
    mock_store.search.return_value = [
        {"doc_id": "doc_1", "content": "Mock document", "metadata": None, "score": 0.95}
    ]
    mock_store.collection_name = "mock_collection"
    mock_store.document_count = 2
    mock_store.get_collection_info.return_value = {
        "name": "mock_collection",
        "count": 2,
    }

    # 使用 Mock 创建客户端
    client = RAGClient(embedding=mock_embedding, vector_store=mock_store)

    # 测试添加文档
    ids = client.add_documents(["Test doc 1", "Test doc 2"])
    print(f"Added documents: {ids}")
    assert mock_embedding.encode.called
    assert mock_store.add_documents.called

    # 测试搜索
    results = client.search("test query")
    print(f"Search results: {len(results)}")
    assert results[0]["score"] == 0.95

    # 获取统计
    stats = client.get_stats()
    print(f"Stats: embedding={stats['embedding_model']}, docs={stats['document_count']}")

    print("\n✅ 示例 4 完成（所有 Mock 调用成功）\n")


def example_5_configuration_serialization():
    """示例 5：配置序列化"""
    print("=" * 60)
    print("示例 5：配置序列化")
    print("=" * 60)

    import json

    # 创建配置
    config = RAGConfig.default_chinese()

    # 序列化为字典
    config_dict = config.to_dict()
    print("配置字典:")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))

    # 保存到文件（示例）
    # with open("rag_config.json", "w") as f:
    #     json.dump(config_dict, f, indent=2)

    # 从字典加载
    loaded_config = RAGConfig.from_dict(config_dict)
    print(f"\n加载的配置: embedding={loaded_config.embedding.model_name}")

    print("\n✅ 示例 5 完成\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG 系统重构示例")
    print("=" * 60 + "\n")

    # 运行所有示例
    example_1_simple_usage()
    example_2_config_based()
    example_3_dependency_injection()
    example_4_testing_with_mocks()
    example_5_configuration_serialization()

    print("=" * 60)
    print("所有示例运行完毕！")
    print("=" * 60)
