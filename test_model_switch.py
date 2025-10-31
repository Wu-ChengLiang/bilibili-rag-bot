#!/usr/bin/env python3
"""Quick test: verify model switching works"""

from src.rag import RAGClient
from src.rag.config import RAGConfig, EmbeddingConfig
from src.rag.embeddings import create_embedding

print("测试 GTE 模型切换\n")

# 测试 1: 通过配置创建
print("1️⃣  通过 RAGConfig 切换到 GTE:")
config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="gte",
        model_name="thenlper/gte-base-zh"
    )
)
client = RAGClient.from_config(config)
print(f"   ✓ 已创建 RAGClient，embedding_model: {client.embedding.model_name}")

# 测试 2: 直接注入
print("\n2️⃣  直接注入 GTE embedding:")
embedding = create_embedding("gte", "thenlper/gte-base-zh")
client = RAGClient(embedding=embedding)
print(f"   ✓ 已创建 RAGClient，embedding_model: {client.embedding.model_name}")

# 测试 3: 获取系统统计
print("\n3️⃣  系统统计信息:")
stats = client.get_stats()
print(f"   - 嵌入模型: {stats['embedding_model']}")
print(f"   - 向量维度: {stats['embedding_dimension']}")
print(f"   - 向量存储: {stats['vector_store']}")

# 测试 4: 简单搜索流程
print("\n4️⃣  测试搜索流程:")
test_doc = "这是一个关于人工智能的测试文档。"
client.add_document(test_doc, metadata={"source": "test"})
print(f"   ✓ 已添加文档")

query = "人工智能"
results = client.search(query, limit=1)
print(f"   ✓ 搜索成功: \"{query}\"")
if results:
    print(f"     - 相似度: {results[0]['score']:.4f}")

client.reset()

print("\n✅ 所有测试通过！GTE 模型已集成成功。")
print("\n使用方式总结:")
print("  • 方式1: RAGConfig + from_config()")
print("  • 方式2: 直接注入 embedding 对象")
print("  • 方式3: 使用工厂函数 create_embedding()")
