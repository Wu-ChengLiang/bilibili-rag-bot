# RAG 交互式对话系统 - 使用指南

## 📌 概览

`main.py` 是一个交互式对话脚本，支持：
- 从**本地文件**（.txt, .md）加载数据
- 从**飞书**实时加载数据（可选）
- 使用 **RAG + LLM** 进行智能问答
- **交互式对话循环**

## 🚀 快速开始

### 1. 环境配置

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 填入你的 API 密钥
# 必填：MOONSHOT_API_KEY
# 可选：FEISHU_APP_ID, FEISHU_APP_SECRET
```

### 2. 准备数据

在 `./docs` 目录下放入你的文档：
```bash
mkdir -p docs
echo "向量数据库是存储和查询高维向量的数据库系统" > docs/vector_db.txt
echo "RAG（检索增强生成）结合了向量搜索和 LLM 能力" > docs/rag.txt
```

### 3. 启动对话

```bash
# 基础使用
python3 main.py

# 或指定 API 密钥
python3 main.py --llm-api-key sk-xxx

# 指定数据目录
python3 main.py --data-dir ./my_docs

# 使用飞书文档
python3 main.py --feishu-doc-ids XCJzwF6Pqi1t5UkUVnpcCSsQnQd another_doc_id

# 同时使用本地和飞书
python3 main.py --data-dir ./docs --feishu-doc-ids doc_id_1 doc_id_2
```

## 📖 使用示例

### 交互式对话

```bash
$ python3 main.py

========================================
🤖 RAG 智能问答系统
========================================
输入你的问题，系统将根据文档回答
输入 'exit' 或 'quit' 退出

你: 什么是向量数据库？

🔍 搜索中...

助手: 向量数据库是存储和查询高维向量的数据库系统...

你: 它有什么优势？

🔍 搜索中...

助手: 向量数据库的优势包括...

你: exit

👋 再见！
```

### Python 代码使用

```python
from main import RAGChatbot

# 初始化
chatbot = RAGChatbot(
    llm_api_key="sk-xxx",
    data_directory="./docs"
)

# 方式 1: 简单对话
response = chatbot.chat("什么是 RAG？")
print(response)

# 方式 2: 对话并返回上下文
result = chatbot.chat_with_context("什么是向量数据库？")
print(f"问题: {result['query']}")
print(f"回答: {result['answer']}")
print(f"相关文档: {result['context']}")

# 方式 3: 仅搜索（不使用 LLM）
search_results = chatbot.search_only("RAG")
for doc in search_results:
    print(f"[{doc['score']:.2f}] {doc['content'][:100]}...")

# 方式 4: 启动交互式对话
chatbot.interactive_chat()

# 查看统计信息
stats = chatbot.get_stats()
print(f"文档数量: {stats['document_count']}")
print(f"向量维度: {stats['embedding_dimension']}")
```

## 🔧 高级用法

### 指定本地和飞书文档

```python
from main import RAGChatbot
from src.data.config import FeishuConfig

# 飞书配置
feishu_config = FeishuConfig(
    app_id="your_app_id",
    app_secret="your_app_secret"
)

# 初始化 chatbot，同时使用本地和飞书文档
chatbot = RAGChatbot(
    llm_api_key="sk-xxx",
    local_directory="./my_docs",
    feishu_doc_ids=["doc_id_1", "doc_id_2"],
    feishu_config=feishu_config
)
```

### 自定义搜索参数

```python
# 搜索并限制相关性阈值
result = chatbot.chat_with_context(
    query="什么是 RAG？",
    limit=5  # 返回前 5 个结果
)
```

### 不使用 LLM（仅向量搜索）

```python
# 如果不提供 llm_api_key，系统仅提供搜索功能
chatbot = RAGChatbot(
    llm_api_key=None,
    data_directory="./docs"
)

# 搜索
results = chatbot.search_only("向量数据库", limit=5)
for doc in results:
    print(f"[{doc['score']:.2f}] {doc['content'][:200]}...")
```

## 📊 系统架构

```
main.py (交互式脚本)
  ├─ RAGChatbot (对话机器人)
  │   ├─ RAGClient (向量搜索)
  │   ├─ LocalFileLoader (本地文件加载)
  │   ├─ FeishuDocxLoader (飞书文档加载)
  │   └─ LLMClient (Kimi API 集成)
  │
  └─ 数据流：
      加载数据 → Embedding → 向量存储 → 搜索 → LLM 生成回答
```

## ⚙️ 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--local-dir` | 本地文件目录 | 无 |
| `--data-dir` | 数据目录（扫描所有 .txt/.md） | `./docs` |
| `--feishu-doc-ids` | 飞书文档 ID 列表 | 无 |
| `--llm-api-key` | LLM API 密钥 | 从 .env 读取 |
| `--llm-model` | LLM 模型名称 | `moonshot-v1-8k` |

## 🧪 运行测试

```bash
# 运行所有测试
python3 -m pytest tests/test_main.py -v

# 运行特定测试
python3 -m pytest tests/test_main.py::TestRAGChatbot::test_chatbot_can_answer_question -v

# 查看覆盖率
python3 -m pytest tests/test_main.py --cov=main
```

## 🐛 故障排查

### 问题：模块未找到

```
ModuleNotFoundError: No module named 'main'
```

**解决方案：** 确保在项目根目录运行脚本
```bash
cd /path/to/rag
python3 main.py
```

### 问题：API 认证失败

```
openai.AuthenticationError: Error code: 401
```

**解决方案：** 检查 MOONSHOT_API_KEY 是否正确
```bash
export MOONSHOT_API_KEY=sk-your_key
python3 main.py
```

### 问题：未加载任何文档

**检查：**
1. 数据目录是否存在
2. 文件是否为 .txt 或 .md 格式
3. 文件是否为空

```bash
ls -la ./docs/
```

### 问题：向量搜索无结果

**原因：** 可能是相似度阈值设置过高
**解决方案：** 降低阈值或增加 `limit` 参数

## 📝 常见问题

### Q: 能否离线使用？
A: 是的！使用 `search_only()` 方法可以进行向量搜索而不依赖 LLM

### Q: 支持哪些文件格式？
A: 当前支持 `.txt` 和 `.md` 文件

### Q: 可以加载 PDF 吗？
A: 暂不支持，但可以先将 PDF 转换为 txt 再加载

### Q: 如何提高搜索准确度？
A:
1. 增加 `limit` 参数（检索更多结果）
2. 优化文档的分块大小
3. 使用更好的 embedding 模型

### Q: 大文件会导致内存溢出吗？
A: ChromaDB 有内存限制，建议单个文件控制在 10MB 以内

## 🔐 隐私和安全

- API 密钥存储在 `.env` 文件中（**不要提交到 Git**）
- 本地向量存储保存在 `./chroma_db` 目录
- 所有通信使用 HTTPS

确保在 `.gitignore` 中包含：
```
.env
chroma_db/
__pycache__/
*.pyc
.pytest_cache/
```

## 📚 相关文档

- [主 README](./README.md) - 完整的 RAG 系统文档
- [CLAUDE.md](./CLAUDE.md) - 项目架构和开发指南
- [Kimi API 文档](https://platform.moonshot.cn/docs)

## 🤝 反馈和改进

有问题或建议？欢迎：
1. 查看 `tests/test_main.py` 中的使用示例
2. 提出 Issue
3. 提交 Pull Request

---

**Happy coding! 🚀**
