# CLAUDE.md

该文件为Claude Code（claude.ai/code）提供了在本项目中处理代码时的指导。

## 📋 项目概览

这是一个**企业级RAG（检索增强生成）系统**，并进行了中文优化。它结合了以下内容：

* **RAG管道**：文档引入 → 嵌入 → 向量搜索 → LLM集成
* **实时飞书集成**：从飞书（Feishu）实时获取文档，并进行定时刷新
* **针尖中的线测试**：RAG检索准确性的综合测试框架
* **长上下文测试**：LLM评估框架（类似于Arize Phoenix）

**核心技术栈：**

* ChromaDB用于向量存储
* Text2Vec用于中文嵌入
* 飞书API用于实时文档加载
* APScheduler用于任务调度
* Pytest用于测试

---

## 🏗️ 高级架构

### 核心RAG系统（`src/rag/`）

**设计模式：** 依赖注入 + 工厂模式 + 策略模式

```
src/rag/
├── core/                  # 抽象层（SOLID原则）
│   ├── embedding.py       # BaseEmbedding ABC - 所有嵌入模型的接口
│   ├── vector_store.py    # BaseVectorStore ABC - 所有向量数据库的接口
│   └── protocols.py       # 协议类型定义
│
├── embeddings/            # 嵌入实现
│   └── text2vec.py        # Text2VecEmbedding（中文优化）
│
├── stores/                # 向量存储实现
│   ├── chromadb_store.py  # ChromaDB实现
│   └── factory.py         # 向量存储的工厂
│
├── client.py              # RAGClient（主要入口，依赖注入）
├── config.py              # 配置管理（基于数据类）
├── types.py               # 类型定义（Document, SearchResult等）
│
├── reranker.py            # 多因素重排序（向量+关键词+长度）
├── llm_client.py          # LLM集成（Kimi/Moonshot API）
├── needle_test.py         # Needle In Haystack测试框架
├── long_context_test.py   # LLM长上下文评估
└── realtime_feishu_rag.py # 实时飞书文档集成
```

### 数据加载与调度（`src/data/`，`src/scheduler.py`）

**数据加载器：** 抽象基类 + 多种实现

* `BaseDataLoader` - 抽象接口
* `FeishuLoader` - 从飞书工作区获取
* `FeishuDocxLoader` - 解析飞书中的DOCX文件
* `LocalFileLoader` - 加载本地文件

**调度器：** 使用APScheduler的`RAGScheduler`用于定期刷新向量存储

---

## 🎯 关键架构原则

### 1. **依赖注入**

* 组件（嵌入、向量存储）是注入的，而不是硬编码的
* 易于模拟，便于测试
* 配置驱动或显式参数注入

```python
# 选项 1：使用默认值（中文优化）
client = RAGClient()

# 选项 2：注入自定义组件
embedding = Text2VecEmbedding(model_name="custom-model")
store = ChromaDBStore(persist_directory="./my_db")
client = RAGClient(embedding=embedding, vector_store=store)

# 选项 3：配置驱动
config = RAGConfig(embedding=..., vector_store=...)
client = RAGClient.from_config(config)
```

### 2. **抽象层（SOLID - DIP）**

* `BaseEmbedding` - 所有嵌入模型继承此类
* `BaseVectorStore` - 所有向量存储继承此类
* 易于添加新提供者（OpenAI，Qdrant等），无需修改现有代码

### 3. **工厂模式**

* `stores/factory.py` - 根据配置创建向量存储
* 集中管理对象创建

### 4. **配置管理**

* 类型安全的数据类：`RAGConfig`，`EmbeddingConfig`，`VectorStoreConfig`
* 可序列化：`to_dict()` / `from_dict()`方法
* 预设配置：`RAGConfig.default_chinese()`

---

## 🚀 常见开发任务

### 运行测试

```bash
# 所有测试
pytest tests/ -v

# 特定测试文件
pytest tests/test_refactored_client.py -v

# 包含覆盖率
pytest --cov=src/rag tests/

# 飞书集成测试
pytest tests/feishu/ -v
```

### 构建与开发

```bash
# 安装依赖
pip install -r requirements.txt

# 检查Python版本（需要3.9+）
python --version

# 运行特定示例
python interactive_search.py
python example_refactored.py

# 运行飞书实时RAG
python -m src.rag.realtime_feishu_rag
```

### 代码质量

```bash
# 检查导入（确保没有循环依赖）
python -c "import src.rag; print('Import OK')"

# 列出目录结构
find src -name "*.py" | head -20
```

---

## 📦 依赖结构

### `src/rag/` 内部依赖

**依赖流（遵循DIP）：**

```
RAGClient
  ├→ BaseEmbedding（抽象）
  ├→ BaseVectorStore（抽象）
  ├→ Reranker（工具）
  ├→ RAGConfig（配置）
  └→ Types（Document, SearchResult）

具体实现：
  Text2VecEmbedding → BaseEmbedding
  ChromaDBStore → BaseVectorStore
```

**无循环依赖** - 使用ABC + 依赖注入模式

### `src/data/` 内部依赖

```
RAGClient → BaseDataLoader
  ├→ FeishuLoader
  ├→ FeishuDocxLoader
  └→ LocalFileLoader

每个加载器都是独立的，无循环依赖
```

### 外部依赖（在`requirements.txt`中）

* `chromadb>=0.4.22` - 向量数据库
* `text2vec>=1.1.5` - 中文嵌入
* `pytest>=7.4.0` - 测试
* `python-dotenv>=1.0.0` - 环境配置
* `apscheduler` - 任务调度（用于实时飞书RAG）

---

## 🔑 核心概念

### 1. **RAGClient**（`src/rag/client.py`）

主要入口。方法：

* `add_document(content, metadata, doc_id)` - 添加单个文档
* `add_documents(documents, metadatas, doc_ids)` - 添加批量文档
* `search(query, limit, score_threshold, filter_metadata)` - 搜索
* `delete_documents(ids)` - 删除文档
* `reset()` - 清除所有文档
* `get_stats()` - 系统统计
* `from_config(config)` - 工厂方法
* `default_chinese()` - 预设配置

### 2. **配置**（`src/rag/config.py`）

* `RAGConfig` - 主要配置，包括嵌入和向量存储设置
* `EmbeddingConfig` - 嵌入模型配置
* `VectorStoreConfig` - 向量存储配置
* 所有配置都为数据类，支持`to_dict()` / `from_dict()`序列化

### 3. **Reranker**（`src/rag/reranker.py`）

多因素重排序：向量相似度 + 关键词匹配 + 长度归一化

* 通过`config.enable_reranking = True`启用
* 在返回给用户之前对搜索结果进行操作

### 4. **BaseEmbedding & BaseVectorStore**（`src/rag/core/`）

定义接口合同的抽象基类：

* 所有新的嵌入/向量存储必须继承自这些类
* 确保与RAGClient兼容

### 5. **实时飞书RAG**（`src/rag/realtime_feishu_rag.py`）

扩展RAGClient，提供：

* 从飞书工作区实时获取文档
* 通过`RAGScheduler`定时刷新
* 自动更新文档

### 6. **测试框架**

* **NeedleTest**（`needle_test.py`） - 测量RAG管道的准确性
* **LongContextTest**（`long_context_test.py`） - 评估LLM的长上下文理解

---

## 🧪 测试策略

### 测试结构

```
tests/
├── test_refactored_client.py    # 主要的RAGClient测试（16个测试，全部通过）
├── test_client.py               # 旧版客户端测试
├── test_needle.py               # Needle In Haystack测试
├── test_document_loader.py      # 文档加载测试
└── feishu/                      # 飞书集成测试
    ├── test_feishu_docx.py
    ├── test_realtime_rag.py
    └── test_debug_blocks.py
```

### 测试方法

* **单元测试**：使用Mock对象注入RAGClient
* **集成测试**：使用真实的ChromaDB + text2vec进行测试
* **功能测试**：测试飞书加载和调度

### 关键测试模式

```python
# 模拟模式（来自test_refactored_client.py）
from unittest.mock import Mock
from src.rag import RAGClient, BaseEmbedding, BaseVectorStore

mock_embedding = Mock(spec=BaseEmbedding)
mock_store = Mock(spec=BaseVectorStore)
client = RAGClient(embedding=mock_embedding, vector_store=mock_store)
```

---

## 📝 项目规范

### 1. **类型提示**

* 所有函数必须有类型提示（Python 3.9+）
* 使用`Optional[T]`表示可为空的值
* 使用`Union[str, List[str]]`表示多态输入

### 2. **文档注释**

* 使用Google风格的文档注释
* 包括参数（Args），返回（Returns），引发异常（Raises）部分
* 在文档注释中添加使用示例

### 3. **错误处理**

* 对无效输入（空内容、不支持的提供者等）抛出`ValueError`
* 记录重要操作（使用`logging`模块）
* 提供有用的错误信息

### 4. **配置**

* 始终支持直接初始化和基于配置的初始化
* 对所有配置对象使用数据类
* 实现`to_dict()` / `from_dict()`进行序列化

### 5. **命名规范**

* 抽象类：`Base<Name>`（例如，`BaseEmbedding`）
* 具体实现：`<Name><Provider>`（例如，`Text2VecEmbedding`，`ChromaDBStore`）
* 测试文件：`test_<module>.py`

---

## 🔄 常见任务工作流

### 添加新的嵌入提供者

1. **创建实现**：在`src/rag/embeddings/<provider>.py`中

   * 继承`BaseEmbedding`
   * 实现`encode()`和`dimension`属性

2. **更新配置**：在`src/rag/config.py`中

   * 将提供者名称添加到`EmbeddingConfig.provider`的Literal中

3. **更新工厂**：在`src/rag/client.py`中

   * 在`_create_embedding_from_config()`中添加新的处理逻辑

4. **添加测试**：在`tests/`中

   * 使用模拟和真实组件进行测试

### 添加新的向量存储

1. **创建实现**：在`src/rag/stores/<provider>.py`中

   * 继承`BaseVectorStore`
   * 实现所有抽象方法

2. **更新工厂**：在`src/rag/stores/factory.py`中

   * 在`create_vector_store()`中添加新的处理逻辑

3. **更新配置**：在`src/rag/config.py`中

   * 将提供者名称添加到`VectorStoreConfig.provider`的Literal中

4. **添加测试**：在`tests/`中

### 添加新的数据加载器

1. **创建实现**：在`src/data/loaders/<loader>.py`中

   * 继承`BaseDataLoader`
   * 实现`load()`方法，返回`List[Document]`

2. **在RAG系统中使用**：通过`RealTimeFeishuRAG`

   * 注册加载器与文档处理器

3. **添加测试**：在`tests/feishu/`中

---

## 🚨 重要架构边界

### 什么内容放在哪里

**`src/rag/`** - 核心RAG逻辑（嵌入、搜索、重排序）

* 无飞书特定的业务逻辑
* 无调度代码

**`src/data/`** - 数据加载抽象

* 为不同源（飞书、本地等）提供加载器
* 文档预处理
* 校验逻辑

**`src/scheduler.py`** - 仅用于调度逻辑

* APScheduler封装
* 不涉及RAG客户端逻辑

**`src/rag/realtime_feishu_rag.py`** - 飞书特定的RAG

* 扩展RAGClient，集成飞书
* 管理刷新逻辑

---

## 🔍 代码导航技巧

### 查找关键实现

* 搜索`class Base`查找所有抽象基类
* 搜索`@abstractmethod`查看接口合同
* 搜索`from_config`查找工厂方法
* 搜索`__init__`并查看类型提示，理解依赖注入

### 理解数据流

1. 用户调用`client.search(query)`
2. RAGClient.search() → `embedding.encode(query)` → `vector_store.search()` → 可选的重排序
3. 返回`List[SearchResult]`

### 调试

* 查看`get_stats()`确认系统状态
* 使用`document_count`属性检查已加载文档
* 启用日志查看详细操作：

  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

---

## 📚 需要了解的关键文件

| 文件                                 | 目的         | 关键类                                                 |
| ---------------------------------- | ---------- | --------------------------------------------------- |
| `src/rag/client.py`                | 主要的RAG客户端  | `RAGClient`                                         |
| `src/rag/config.py`                | 配置管理       | `RAGConfig`, `EmbeddingConfig`, `VectorStoreConfig` |
| `src/rag/core/embedding.py`        | 嵌入接口       | `BaseEmbedding`                                     |
| `src/rag/core/vector_store.py`     | 向量存储接口     | `BaseVectorStore`                                   |
| `src/rag/embeddings/text2vec.py`   | 中文嵌入实现     | `Text2VecEmbedding`                                 |
| `src/rag/stores/chromadb_store.py` | ChromaDB实现 | `ChromaDBStore`                                     |
| `src/rag/reranker.py`              | 重排序逻辑      | `Reranker`                                          |
| `src/rag/realtime_feishu_rag.py`   | 飞书集成       | `RealTimeFeishuRAG`                                 |
| `src/scheduler.py`                 | 任务调度       | `RAGScheduler`                                      |
| `src/data/loaders/base.py`         | 加载器接口      | `BaseDataLoader`                                    |

---

## 🎓 使用的设计模式

1. **依赖注入** - 组件注入到RAGClient中
2. **工厂模式** - `create_vector_store()`，`_create_embedding_from_config()`
3. **策略模式** - 多种可互换的嵌入/存储实现
4. **抽象基类（ABC）** - `BaseEmbedding`，`BaseVectorStore`，`BaseDataLoader`
5. **配置对象** - `RAGConfig`与嵌套配置
6. **构建者模式** - `RAGConfig.default_chinese()`，`RAGClient.from_config()`

---

## 🧠 需要记住的事项

1. **始终遵守抽象接口** - 不要绕过`BaseEmbedding`/`BaseVectorStore`
2. **配置是首类公民** - 使用配置对象，而不是散布的参数
3. **类型提示非常重要** - 它们启用IDE自动补全并提前捕获错误
4. **首先使用模拟进行测试** - 比集成测试反馈更快
5. **记录假设** - 尤其是关于数据格式和API合同的假设
6. **保持依赖图干净** - 不要有循环依赖，避免不必要的导入
7. **飞书集成是可选的** - 核心RAG即使没有飞书也能工作；保持关注点分离

---

## 🚀 API 服务启动

### 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 FastAPI 服务
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 3. 访问 API
# 浏览器打开: http://localhost:8000/docs
# 或通过 curl 调用
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "test",
    "user_id": "user_001",
    "user_name": "用户",
    "message": "你好"
  }'
```

### 环境配置

在 `.env` 文件中配置（项目已包含）：

```env
# 选择 LLM 提供者
MOONSHOT_API_KEY=sk_xxxxx          # Kimi/Moonshot API
ZHIPU_API_KEY=xxxxx                # Zhipu GLM API

# 数据和历史
DATA_DIRECTORY=./data              # 文档数据目录
HISTORY_DIR=./history              # 对话历史目录

# Feishu 集成（可选）
FEISHU_APP_ID=xxxxx
FEISHU_APP_SECRET=xxxxx
FEISHU_WIKI_SPACE_ID=xxxxx
```

### API 端点

| 端点 | 方法 | 功能 | 请求体 |
|------|------|------|--------|
| `/health` | GET | 健康检查 | - |
| `/stats` | GET | 系统统计 | - |
| `/chat` | POST | 多轮对话 | ChatRequest |
| `/clear-history` | POST | 清空历史 | platform, user_id |

### 示例：ChatRequest

```json
{
  "platform": "bilibili",
  "user_id": "123456",
  "user_name": "用户名",
  "message": "用户问题",
  "history": [
    {"role": "user", "content": "之前的问题"},
    {"role": "assistant", "content": "之前的回答"}
  ]
}
```

### 示例：ChatResponse

```json
{
  "success": true,
  "reply": "回答内容"
}
```

### LLM 提供者选择

#### 使用 Kimi（默认）

```bash
# api/main.py 中配置
llm_provider = "kimi"  # 或 "moonshot"
llm_api_key = os.getenv("MOONSHOT_API_KEY")
```

#### 使用 Zhipu GLM

```bash
# api/main.py 中配置
llm_provider = "zhipu"  # 或 "glm"
llm_api_key = os.getenv("ZHIPU_API_KEY")
```

### 可用的 LLM 提供者

```python
from src.llm.factory import LLMFactory

# 列出所有可用提供者
providers = LLMFactory.list_providers()
# ['kimi', 'moonshot', 'zhipu', 'glm']

# 创建客户端
client = LLMFactory.create(
    provider="kimi",
    api_key=os.getenv("MOONSHOT_API_KEY"),
    model="moonshot-v1-8k"
)
```

### Embedding 模型

目前支持：

- **Text2VecEmbedding** - 轻量级，速度快（text2vec-base-chinese）
- **GTEEmbedding** - 平衡方案，精度好（gte-base-zh）

### System Prompt 管理

所有 LLM 的 system prompt 使用 **Jinja2 模板** 管理，存储在 `src/llm/prompts/` 目录：

- `system_single.jinja2` - 单轮对话
- `system_multi.jinja2` - 多轮对话

**特性**：
- 所有回答控制在 450 字以内
- 模板与代码分离，易于维护
- 所有 LLM 实现共享相同的 prompt

### 常见问题

**Q: 模型 loading 很慢？**
A: 首次使用时会下载模型文件，建议在后台等待或使用 `-q` 参数减少日志输出

**Q: 如何更换 LLM？**
A: 修改 `api/main.py` 中的 `llm_provider` 和 `llm_api_key`，然后重启服务

**Q: 如何自定义 prompt？**
A: 编辑 `src/llm/prompts/*.jinja2` 文件，不需要改代码，重启服务即生效
