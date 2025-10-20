# RAG Needle In a Haystack Test

MVP implementation of a RAG system with Chinese support and Needle In a Haystack testing.

## Features

- **Vector Database**: ChromaDB with persistent storage
- **Embedding**: Text2Vec for Chinese text
- **Testing**: Needle In a Haystack evaluation framework
- **Development**: TDD approach

## Architecture

```
src/rag/
├── client.py          # ChromaDB client with Text2Vec
├── types.py           # Type definitions
└── needle_test.py     # Needle In a Haystack test framework

tests/
├── test_client.py     # Client tests
├── test_search.py     # Search functionality tests
└── test_needle.py     # Needle In a Haystack tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.rag.client import RAGClient

# Initialize client
client = RAGClient(persist_directory="./chroma_db")

# Add documents
client.add_documents(["文档1", "文档2"])

# Search
results = client.search("查询文本", limit=5)
```

## Testing

```bash
pytest tests/
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector DB | ChromaDB |
| Embedding | Text2Vec |
| Persistence | File system |
| Testing | pytest |
