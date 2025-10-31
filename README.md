# RAG System with BiliGo Integration

一个企业级的 RAG（Retrieval-Augmented Generation）系统，集成了 B站私信 AI 自动回复功能，具备完整的架构设计、中文优化和多用户对话管理。

## ✨ 特性

- 🏗️ **企业级架构** - 依赖注入、SOLID 原则、易于扩展
- 🇨🇳 **中文优化** - 智能文本分块、多种中文 embedding、重排序
- 🔌 **可扩展** - 支持多种 embedding (text2vec, GTE) 和向量数据库
- 🤖 **BiliGo 集成** - B站私信 AI 自动回复系统（基于 RAG）
- 💬 **对话管理** - 多用户对话历史管理和持久化
- 🎯 **多 LLM 支持** - Kimi/Moonshot 和智谱 GLM API

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

## 📊 核心功能

### RAG 搜索与检索

完整的文档加载、向量化和智能搜索：

```python
from src.rag import RAGClient

client = RAGClient()

# 自动从 ./data 目录加载文档
# 支持 .md, .txt 等文本格式

# 搜索
results = client.search("查询内容", limit=5)

for result in results:
    print(f"相关度: {result['score']:.2f}")
    print(f"内容: {result['content']}")
```

### 对话管理

支持多用户的对话历史管理：

```python
# 对话历史自动保存到 ./history/{platform}/{user_id}.json
# 用户重新交互时，自动加载之前的对话历史
```

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
| **RAGClient** | 主 RAG 客户端 | 文档管理、搜索、配置管理 |
| **Reranker** | 智能重排序 | 多因子评分（向量+关键词+长度） |
| **Embeddings** | 嵌入模型 | Text2Vec、GTE (中文优化) |
| **LLMClient** | LLM 集成 | 支持 Kimi、智谱 GLM |
| **ConversationManager** | 对话管理 | 多用户历史管理和持久化 |

## 📈 运行示例

### 启动 RAG 服务并测试

```bash
# 启动服务
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# 测试 API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "test",
    "user_id": "user_001",
    "user_name": "测试用户",
    "message": "你好"
  }'
```

### B站私信集成示例

```bash
# 启动 BiliGo
cd BiliGo && python3 app.py &

# 在 Web 界面配置 B站凭证
# http://localhost:4999
# 启动私信监控后，系统会自动回复消息
```

## 📚 文档

- [CLAUDE.md](./CLAUDE.md) - 项目开发指南和最佳实践
- [MAIN_USAGE.md](./MAIN_USAGE.md) - 主程序使用说明

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

# LLM 模块测试
pytest tests/test_llm_module.py -v

# 飞书集成测试
pytest tests/feishu/ -v
```

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
