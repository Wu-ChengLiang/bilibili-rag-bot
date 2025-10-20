# RAG 系统重构迁移指南

## 概述

RAG 系统已完成架构重构（v2.0.0），解决了以下核心问题：

- ✅ **无抽象层** → 引入 `BaseEmbedding` 和 `BaseVectorStore` 抽象基类
- ✅ **硬编码依赖** → 使用依赖注入，支持灵活配置
- ✅ **违反 DIP** → 依赖抽象接口而非具体实现
- ✅ **单一实现路径** → 支持多种 embedding 和向量数据库（可扩展）
- ✅ **难以测试** → 可注入 Mock 对象，易于单元测试

## 新架构结构

```
src/rag/
├── core/                   # 抽象层
│   ├── embedding.py        # BaseEmbedding 抽象基类
│   ├── vector_store.py     # BaseVectorStore 抽象基类
│   └── protocols.py        # Protocol 定义
├── embeddings/             # Embedding 实现
│   └── text2vec.py         # Text2Vec 实现
├── stores/                 # VectorStore 实现
│   ├── chromadb_store.py   # ChromaDB 实现
│   └── factory.py          # 工厂方法
├── config.py               # 配置管理
├── client.py               # 重构后的 RAGClient
└── client_legacy.py        # 旧版本（向后兼容）
```

## 迁移步骤

### 1. 简单迁移（零代码更改）

旧代码**无需修改**即可工作：

```python
# 旧代码（仍然有效）
from src.rag import RAGClient

client = RAGClient()  # 使用默认中文配置
client.add_documents(["文档1", "文档2"])
results = client.search("查询")
```

### 2. 推荐的新用法

#### 方式 1：使用配置对象

```python
from src.rag import RAGClient, RAGConfig

# 使用默认中文优化配置
config = RAGConfig.default_chinese()
client = RAGClient.from_config(config)

# 或自定义配置
config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="text2vec",
        model_name="shibing624/text2vec-base-chinese"
    ),
    vector_store=VectorStoreConfig(
        provider="chromadb",
        persist_directory="./my_db"
    ),
    enable_reranking=True
)
client = RAGClient.from_config(config)
```

#### 方式 2：依赖注入（用于测试）

```python
from src.rag import RAGClient
from src.rag.embeddings import Text2VecEmbedding
from src.rag.stores import ChromaDBStore

# 手动创建依赖
embedding = Text2VecEmbedding(model_name="your-model")
store = ChromaDBStore(persist_directory="./db")

# 注入依赖
client = RAGClient(embedding=embedding, vector_store=store)
```

#### 方式 3：Mock 测试

```python
from unittest.mock import Mock
from src.rag import RAGClient, BaseEmbedding, BaseVectorStore

# 创建 Mock 对象
mock_embedding = Mock(spec=BaseEmbedding)
mock_store = Mock(spec=BaseVectorStore)

# 注入 Mock（完美的单元测试）
client = RAGClient(embedding=mock_embedding, vector_store=mock_store)
```

## API 变化

### RAGClient

#### 新增方法

```python
# 工厂方法
client = RAGClient.from_config(config)
client = RAGClient.default_chinese()

# 新增功能
stats = client.get_stats()  # 获取系统统计信息
client.delete_documents(ids)  # 删除文档
```

#### 增强的方法

```python
# search 支持元数据过滤
results = client.search(
    query="查询",
    limit=10,
    score_threshold=0.7,
    filter_metadata={"source": "book"}  # 新增
)
```

### 配置系统

```python
from src.rag import RAGConfig, EmbeddingConfig, VectorStoreConfig

# 配置序列化
config = RAGConfig.default_chinese()
config_dict = config.to_dict()

# 配置反序列化
config = RAGConfig.from_dict(config_dict)
```

## 扩展性示例

### 添加新的 Embedding 提供商

```python
# 1. 实现 BaseEmbedding
from src.rag.core.embedding import BaseEmbedding

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key, model="text-embedding-3-small"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def encode(self, texts):
        response = self._client.embeddings.create(
            input=texts, model=self._model
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self):
        return 1536  # text-embedding-3-small 维度

# 2. 使用新 Embedding
embedding = OpenAIEmbedding(api_key="sk-xxx")
client = RAGClient(embedding=embedding)
```

### 添加新的向量数据库

```python
# 1. 实现 BaseVectorStore
from src.rag.core.vector_store import BaseVectorStore

class QdrantStore(BaseVectorStore):
    def __init__(self, url, api_key):
        # 实现 Qdrant 客户端
        pass

    def add_documents(self, documents, embeddings, ...):
        # 实现添加逻辑
        pass

    def search(self, query_embedding, ...):
        # 实现搜索逻辑
        pass

    # ... 实现其他抽象方法

# 2. 使用新 VectorStore
store = QdrantStore(url="http://localhost:6333", api_key="xxx")
client = RAGClient(vector_store=store)
```

## 性能提升

### 批处理优化

```python
# 新架构支持批处理
embedding = Text2VecEmbedding()
embeddings = embedding.encode_batch(large_doc_list, batch_size=64)
```

### Reranking

```python
# 启用 reranking 提升检索质量
config = RAGConfig.default_chinese()
config.enable_reranking = True

client = RAGClient.from_config(config)
```

## 测试改进

### 旧版本（难以测试）

```python
# 必须使用真实的 text2vec 和 ChromaDB
client = RAGClient()  # 初始化缓慢，依赖文件系统
results = client.search("test")  # 难以预测结果
```

### 新版本（易于测试）

```python
# 可以完全隔离测试
from unittest.mock import Mock

mock_embedding = Mock()
mock_embedding.encode.return_value = [0.1] * 384

mock_store = Mock()
mock_store.search.return_value = [{"doc_id": "1", "score": 0.9}]

client = RAGClient(embedding=mock_embedding, vector_store=mock_store)
results = client.search("test")  # 完全可控的测试
```

## 向后兼容性

旧代码可以继续使用 `client_legacy.py`：

```python
from src.rag.client_legacy import RAGClient as LegacyRAGClient

# 旧代码无需修改
client = LegacyRAGClient(
    persist_directory="./chroma_db",
    model_name="shibing624/text2vec-base-chinese"
)
```

## 总结

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 依赖管理 | 硬编码 | 依赖注入 |
| 可测试性 | 困难 | 容易（可 Mock） |
| 扩展性 | 需修改源码 | 实现接口即可 |
| 配置管理 | 散落参数 | 统一配置对象 |
| SOLID 原则 | 违反多个 | 完全遵守 |
| 向后兼容 | N/A | 100% 兼容 |

**建议：新项目使用新架构，旧项目逐步迁移。**
