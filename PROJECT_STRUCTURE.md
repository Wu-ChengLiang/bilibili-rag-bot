# RAG 系统项目结构 (v2.0)

## 📁 完整目录结构

```
rag/
├── src/rag/                       # 核心模块
│   ├── core/                      # 🆕 抽象层
│   │   ├── __init__.py
│   │   ├── embedding.py           # BaseEmbedding 抽象基类
│   │   ├── vector_store.py        # BaseVectorStore 抽象基类
│   │   └── protocols.py           # Protocol 类型定义
│   │
│   ├── embeddings/                # 🆕 Embedding 实现
│   │   ├── __init__.py
│   │   └── text2vec.py            # Text2Vec 中文 embedding
│   │
│   ├── stores/                    # 🆕 VectorStore 实现
│   │   ├── __init__.py
│   │   ├── chromadb_store.py      # ChromaDB 向量存储
│   │   └── factory.py             # 工厂方法
│   │
│   ├── __init__.py                # 模块导出
│   ├── client.py                  # ✨ 重构的 RAGClient
│   ├── config.py                  # 🆕 配置管理
│   ├── types.py                   # 类型定义
│   │
│   ├── document_loader.py         # 文档加载与分块
│   ├── reranker.py                # 重排序模块
│   ├── llm_client.py              # LLM 集成
│   ├── needle_test.py             # Needle in Haystack 测试
│   └── long_context_test.py       # 长文本测试
│
├── tests/                         # 测试套件
│   ├── __init__.py
│   ├── test_client.py             # 原有测试
│   ├── test_refactored_client.py  # 🆕 重构后的单元测试
│   ├── test_needle.py
│   └── test_document_loader.py
│
├── example_refactored.py          # 🆕 新架构演示脚本
│
├── ARCHITECTURE_REFACTOR.md       # 🆕 架构重构文档
├── MIGRATION_GUIDE.md             # 🆕 迁移指南
├── PROJECT_STRUCTURE.md           # 🆕 本文档
├── README.md
└── requirements.txt
```

---

## 📦 模块说明

### 🎯 核心模块 (src/rag/)

| 模块 | 说明 | 状态 |
|------|------|------|
| `__init__.py` | 统一导出接口 | ✨ 更新 |
| `client.py` | RAGClient 主类（依赖注入） | ✨ 重构 |
| `config.py` | 配置管理（RAGConfig, EmbeddingConfig, VectorStoreConfig） | 🆕 新增 |
| `types.py` | 类型定义（Document, SearchResult） | ✅ 保留 |

### 🧩 抽象层 (src/rag/core/)

| 模块 | 说明 | 行数 |
|------|------|------|
| `embedding.py` | BaseEmbedding 抽象基类 | ~60 |
| `vector_store.py` | BaseVectorStore 抽象基类 | ~80 |
| `protocols.py` | Protocol 类型协议 | ~50 |

**职责：** 定义统一接口，实现依赖倒置原则

### 🔌 实现层

#### Embeddings (src/rag/embeddings/)

| 模块 | 说明 | 提供商 |
|------|------|--------|
| `text2vec.py` | Text2Vec embedding（中文优化） | text2vec |

**扩展性：** 可添加 OpenAI, Cohere, HuggingFace 等实现

#### Stores (src/rag/stores/)

| 模块 | 说明 | 提供商 |
|------|------|--------|
| `chromadb_store.py` | ChromaDB 向量存储 | ChromaDB |
| `factory.py` | 工厂方法创建 VectorStore | - |

**扩展性：** 可添加 Qdrant, Weaviate, Pinecone 等实现

### 🛠️ 工具模块

| 模块 | 说明 | 特色 |
|------|------|------|
| `document_loader.py` | 文档加载与分块 | 3种策略（sentences, fixed_size, smart） |
| `reranker.py` | 重排序模块 | 多因子评分（向量+关键词+长度） |
| `llm_client.py` | LLM 集成 | Kimi/Moonshot API |
| `needle_test.py` | Needle测试框架 | 检索准确性评估 |

---

## 🔧 使用方式

### 方式 1：默认使用（最简单）

```python
from src.rag import RAGClient

client = RAGClient()  # 自动使用中文优化配置
client.add_documents(["文档1", "文档2"])
results = client.search("查询")
```

### 方式 2：配置驱动

```python
from src.rag import RAGClient, RAGConfig

config = RAGConfig.default_chinese()
config.enable_reranking = True

client = RAGClient.from_config(config)
```

### 方式 3：依赖注入

```python
from src.rag import RAGClient
from src.rag.embeddings import Text2VecEmbedding
from src.rag.stores import ChromaDBStore

embedding = Text2VecEmbedding()
store = ChromaDBStore(persist_directory="./my_db")

client = RAGClient(embedding=embedding, vector_store=store)
```

### 方式 4：测试（Mock）

```python
from unittest.mock import Mock
from src.rag import RAGClient, BaseEmbedding, BaseVectorStore

mock_embedding = Mock(spec=BaseEmbedding)
mock_store = Mock(spec=BaseVectorStore)

client = RAGClient(embedding=mock_embedding, vector_store=mock_store)
```

---

## 🧪 测试覆盖

### 测试文件

| 文件 | 测试内容 | 状态 |
|------|---------|------|
| `test_refactored_client.py` | 新架构完整测试（16个测试用例） | ✅ 16/16 通过 |
| `test_client.py` | 原有功能测试 | ✅ 保留 |
| `test_needle.py` | Needle测试框架 | ✅ 保留 |
| `test_document_loader.py` | 文档加载测试 | ✅ 保留 |

### 测试运行

```bash
# 运行所有测试
python3 -m pytest tests/ -v

# 运行重构测试
python3 -m pytest tests/test_refactored_client.py -v

# 查看覆盖率
python3 -m pytest --cov=src/rag tests/
```

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| `README.md` | 项目概述 |
| `ARCHITECTURE_REFACTOR.md` | 架构重构详细说明 |
| `MIGRATION_GUIDE.md` | 迁移指南 |
| `PROJECT_STRUCTURE.md` | 项目结构（本文档） |

---

## 🎯 设计原则

### SOLID 原则

- ✅ **单一职责** - 每个类只负责一件事
- ✅ **开闭原则** - 对扩展开放，对修改封闭
- ✅ **里氏替换** - 可替换任意实现
- ✅ **接口隔离** - Protocol/ABC 定义清晰
- ✅ **依赖倒置** - 依赖抽象而非具体

### 设计模式

- 🏭 **工厂模式** - `stores/factory.py`
- 💉 **依赖注入** - `RAGClient.__init__`
- 📋 **策略模式** - `BaseEmbedding`, `BaseVectorStore`
- 🔧 **建造者模式** - `RAGConfig.default_chinese()`

---

## 🚀 扩展指南

### 添加新的 Embedding

```python
# 1. 实现 BaseEmbedding
from src.rag.core.embedding import BaseEmbedding

class OpenAIEmbedding(BaseEmbedding):
    def encode(self, texts):
        # 实现逻辑
        pass

    @property
    def dimension(self):
        return 1536

# 2. 使用
embedding = OpenAIEmbedding()
client = RAGClient(embedding=embedding)
```

### 添加新的 VectorStore

```python
# 1. 实现 BaseVectorStore
from src.rag.core.vector_store import BaseVectorStore

class QdrantStore(BaseVectorStore):
    def add_documents(self, ...):
        # 实现逻辑
        pass

    def search(self, ...):
        # 实现逻辑
        pass

# 2. 更新工厂方法
# 在 stores/factory.py 添加
if config.provider == "qdrant":
    return QdrantStore(...)

# 3. 使用
store = QdrantStore()
client = RAGClient(vector_store=store)
```

---

## 📊 代码统计

| 分类 | 文件数 | 代码行数 |
|------|--------|---------|
| 核心抽象 | 3 | ~190 |
| 实现层 | 4 | ~420 |
| 配置管理 | 1 | ~110 |
| 主客户端 | 1 | ~280 |
| 工具模块 | 4 | ~550 |
| 测试 | 4 | ~450 |
| **总计** | **17** | **~2000** |

---

## 🎉 重构成果

### 解决的问题

✅ 无抽象层 → 引入 BaseEmbedding + BaseVectorStore
✅ 硬编码依赖 → 依赖注入
✅ 违反 DIP → 依赖抽象接口
✅ 单一实现 → 支持多种实现
✅ 难以测试 → 可注入 Mock

### 保留的优势

✅ DocumentLoader - 智能中文分块
✅ Reranker - 多因子重排序
✅ NeedleTest - 检索准确性测试
✅ LLMClient - 端到端 RAG 集成

### 架构升级

**v1.0 → v2.0**
单体耦合 → 模块化解耦
硬编码 → 配置驱动
难测试 → 易测试
难扩展 → 易扩展

---

## 📧 联系与贡献

欢迎贡献新的 Embedding 或 VectorStore 实现！

**重构完成！** 🎊
