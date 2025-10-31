# RAG System with Needle In Haystack Testing

一个企业级的 RAG（Retrieval-Augmented Generation）系统，具备完整的架构设计、测试框架和中文优化。

## ✨ 特性

- 🏗️ **企业级架构** - 依赖注入、SOLID 原则、易于扩展
- 🧪 **双重测试框架** - RAG 检索测试 + LLM 长上下文测试
- 🇨🇳 **中文优化** - 智能文本分块、中文 embedding、重排序
- 🔌 **可扩展** - 支持多种 embedding 和向量数据库
- ✅ **完整测试** - 16 个单元测试全部通过
- 🤖 **BiliGo 集成** - B站私信 AI 自动回复系统（基于 RAG）

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 启动 RAG API 服务

```bash
# 启动 FastAPI 服务（端口 8000）
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 验证服务
curl http://localhost:8000/health
```

### 启动 BiliGo B站私信自动回复

```bash
# 进入 BiliGo 子模块目录
cd BiliGo

# 启动 Flask Web 应用（端口 4999）
python3 app.py

# 访问 Web 界面
# 浏览器打开: http://localhost:4999
```

**系统架构：**
- FastAPI RAG 服务 (8000) - 提供 RAG 检索和 AI 回复
- Flask BiliGo (4999) - 私信监控和配置管理

### 基础使用

```python
from src.rag import RAGClient

# 创建客户端（使用中文优化配置）
client = RAGClient()

# 添加文档
docs = ["人工智能是计算机科学的一个分支", "机器学习是人工智能的核心技术"]
client.add_documents(docs)

# 搜索
results = client.search("什么是人工智能？", limit=5)

for result in results:
    print(f"[{result['score']:.2f}] {result['content']}")
```

### 高级用法

```python
from src.rag import RAGClient, RAGConfig

# 自定义配置
config = RAGConfig.default_chinese()
config.enable_reranking = True
config.default_search_limit = 10

client = RAGClient.from_config(config)
```

## 📊 测试框架

### 1. RAG 检索测试（完整 Pipeline）

测试向量检索的准确性：

```python
from src.rag import RAGClient, NeedleTest

client = RAGClient()
tester = NeedleTest(client)

# 运行测试
result = tester.run_test(
    needle="重要信息：宝藏在山顶",
    haystack_size=100,
    query="宝藏在哪里？"
)

print(f"成功: {result['success']}")
print(f"排名: {result['needle_rank']}")
```

**测试内容：**
- ✅ Embedding 模型的语义理解
- ✅ Vector Search 的准确性
- ✅ Reranker 的效果

### 2. LLM 长上下文测试（类似 Arize）

测试 LLM 从长文档中提取信息的能力：

```python
from src.rag import LongContextTest

tester = LongContextTest(api_key="your-kimi-key")

# 运行测试
results = tester.run_comprehensive_test(
    context_lengths=[1000, 5000, 10000],
    needle_positions=["beginning", "middle", "end"],
    trials_per_config=3
)

# 可视化
tester.visualize_results(results)
```

**测试内容：**
- ✅ LLM 长上下文理解能力
- ✅ 不同位置的信息检索
- ✅ 上下文长度影响

## 🏗️ 架构设计

### 核心抽象层

```
src/rag/
├── core/                      # 抽象层
│   ├── embedding.py           # BaseEmbedding 抽象基类
│   ├── vector_store.py        # BaseVectorStore 抽象基类
│   └── protocols.py           # Protocol 定义
├── embeddings/                # Embedding 实现
│   └── text2vec.py            # Text2Vec（中文优化）
├── stores/                    # VectorStore 实现
│   ├── chromadb_store.py      # ChromaDB 实现
│   └── factory.py             # 工厂模式
└── config.py                  # 配置管理
```

### 依赖注入示例

```python
from unittest.mock import Mock
from src.rag import RAGClient, BaseEmbedding, BaseVectorStore

# 注入 Mock 对象（易于测试）
mock_embedding = Mock(spec=BaseEmbedding)
mock_store = Mock(spec=BaseVectorStore)

client = RAGClient(embedding=mock_embedding, vector_store=mock_store)
```

## 🔧 核心组件

| 组件 | 说明 | 特色 |
|------|------|------|
| **DocumentLoader** | 文档加载与分块 | 3种策略：sentences, fixed_size, smart |
| **Reranker** | 重排序模块 | 多因子评分（向量+关键词+长度） |
| **NeedleTest** | RAG检索测试 | Needle In Haystack 方法论 |
| **LongContextTest** | LLM长上下文测试 | 类似 Arize Phoenix |
| **LLMClient** | LLM 集成 | 支持 Kimi/Moonshot API |

## 📈 测试结果

### 单元测试

```bash
pytest tests/ -v
```

```
✅ 16/16 测试通过
- 依赖注入测试
- CRUD 操作测试
- 配置管理测试
- Reranking 测试
- 统计信息测试
```

### Needle In Haystack 测试

**RAG 检索测试结果：**
- 小规模文档（< 50）：✅ 成功率 > 90%
- 中等规模（50-200）：⚠️ 成功率约 60%
- 大规模文档（> 200）：❌ 需要优化

**LLM 长上下文测试结果（Kimi K2）：**
- 短上下文（< 2K）：✅ 100% 准确
- 中等上下文（2K-8K）：✅ 95%+ 准确
- 长上下文（8K-128K）：✅ 90%+ 准确
- **无明显"中间盲点"问题**

## 🎯 两种测试的区别

| 测试类型 | RAG 检索测试 | LLM 长上下文测试 |
|---------|-------------|----------------|
| **测试对象** | 整个 RAG Pipeline | LLM 理解能力 |
| **技术栈** | Embedding + VectorDB + LLM | 仅 LLM |
| **测试流程** | 查询 → 向量检索 → 重排序 → LLM | 直接给 LLM 长文档 |
| **评估指标** | Needle 是否排名第一 | LLM 是否正确回答 |
| **参考项目** | 自研 | Arize Phoenix NIAH |

## 📚 文档

- [架构重构说明](./ARCHITECTURE_REFACTOR.md) - 详细的架构设计
- [迁移指南](./MIGRATION_GUIDE.md) - 从旧版本迁移
- [项目结构](./PROJECT_STRUCTURE.md) - 完整的项目结构

## 🔄 扩展性

### 添加新的 Embedding

```python
from src.rag.core.embedding import BaseEmbedding

class OpenAIEmbedding(BaseEmbedding):
    def encode(self, texts):
        # 实现 OpenAI embedding
        return self.client.embeddings.create(input=texts)

    @property
    def dimension(self):
        return 1536

# 使用
embedding = OpenAIEmbedding(api_key="sk-xxx")
client = RAGClient(embedding=embedding)
```

### 添加新的向量数据库

```python
from src.rag.core.vector_store import BaseVectorStore

class QdrantStore(BaseVectorStore):
    def add_documents(self, documents, embeddings, ...):
        # 实现 Qdrant 逻辑
        pass

    def search(self, query_embedding, ...):
        # 实现搜索逻辑
        pass

# 使用
store = QdrantStore(url="http://localhost:6333")
client = RAGClient(vector_store=store)
```

## 🛠️ 开发

### 运行测试

```bash
# 所有测试
pytest tests/ -v

# 重构后的测试
pytest tests/test_refactored_client.py -v

# RAG 检索测试示例
python rag_demo_full.py

# 长上下文测试示例
python long_context_demo.py
```

### 示例脚本

```bash
# 查看所有演示示例
python example_refactored.py
```

包含 5 个示例：
1. 简单用法
2. 配置驱动
3. 依赖注入
4. Mock 测试
5. 配置序列化

### 完整系统启动

```bash
# 1. 启动 RAG 服务（必需）
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# 2. 启动 BiliGo B站私信系统（可选）
cd BiliGo && python3 app.py &

# 3. 查看 RAG 文档统计
curl http://localhost:8000/stats | python3 -m json.tool

# 4. 查看 BiliGo 监控状态
curl http://localhost:4999/api/status | python3 -m json.tool
```

## 📊 性能指标

- **Embedding 速度**: ~100 docs/s（text2vec-base-chinese）
- **检索延迟**: < 100ms（1000 文档）
- **Reranking 开销**: ~50ms（前10个结果）
- **内存占用**: ~200MB（基础配置）

## 🔑 核心优势

### vs crewAI RAG

✅ **我们的优势：**
- DocumentLoader - 智能中文文本分块
- Reranker - 多因子检索优化
- NeedleTest - 完整的测试框架
- LLMClient - 端到端 RAG 集成

✅ **借鉴 crewAI：**
- 工厂模式设计
- 依赖注入架构
- 配置管理系统
- 抽象层设计

### vs Arize Phoenix

✅ **我们实现：**
- RAG 完整 Pipeline 测试
- LLM 长上下文测试（类似 Arize）
- 中文场景优化
- 可视化结果

## 📝 版本历史

### v2.0.0 (Current)
- ✨ 架构重构：引入抽象层和依赖注入
- ✨ 配置管理：类型安全的配置系统
- ✨ 完整测试：16 个单元测试
- ✨ 长上下文测试：新增 LongContextTest
- 📚 完善文档：架构说明、迁移指南

### v1.0.0
- 🎉 初始版本：基础 RAG 功能
- 🇨🇳 中文优化：DocumentLoader, Reranker
- 🧪 测试框架：NeedleTest

## 📄 License

MIT License

## 🤝 贡献

欢迎贡献新的 Embedding 或 VectorStore 实现！

---

**Built with ❤️ using Claude Code**
