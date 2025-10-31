"""测试 debug_search 的 None 值问题"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag_chat_service import RAGChatService
from unittest.mock import Mock, MagicMock


def test_debug_search_handles_none_results(tmp_path, capsys):
    """测试：debug_search 能正确处理 None 结果"""

    # Mock RAGClient
    mock_rag_client = MagicMock()
    # 模拟返回包含 None 的结果
    mock_rag_client.search.return_value = [
        {
            "score": 0.8,
            "content": "有效内容1",
            "metadata": {"doc_id": "doc1", "title": "标题1"}
        },
        None,  # 这是 None 值
        {
            "score": 0.7,
            "content": "有效内容2",
            "metadata": None,  # metadata 是 None
        },
        {
            "score": 0.6,
            "content": "有效内容3",
            # 没有 metadata 键
        },
    ]

    service = RAGChatService(
        llm_provider="kimi",
        llm_api_key="test_key",
        history_dir=str(tmp_path),
    )
    service.rag_client = mock_rag_client

    # 调用 debug_search，应该不报错
    service.debug_search("何冰", limit=5)

    # 检查输出
    captured = capsys.readouterr()
    output = captured.out

    # 应该有搜索过程的输出
    assert "DEBUG: 搜索过程分析" in output
    assert "Query: 何冰" in output
    # 应该没有异常
    assert "NoneType" not in output


def test_debug_search_with_all_none_results(tmp_path, capsys):
    """测试：debug_search 能处理全是 None 的结果"""

    mock_rag_client = MagicMock()
    mock_rag_client.search.return_value = [None, None, None]

    service = RAGChatService(
        llm_provider="kimi",
        llm_api_key="test_key",
        history_dir=str(tmp_path),
    )
    service.rag_client = mock_rag_client

    # 应该不报错
    service.debug_search("查询", limit=5)

    captured = capsys.readouterr()
    output = captured.out

    assert "DEBUG: 搜索过程分析" in output
    # 应该没有异常
    assert "Error" not in output


def test_debug_search_with_empty_results(tmp_path, capsys):
    """测试：debug_search 能处理空结果"""

    mock_rag_client = MagicMock()
    mock_rag_client.search.return_value = []

    service = RAGChatService(
        llm_provider="kimi",
        llm_api_key="test_key",
        history_dir=str(tmp_path),
    )
    service.rag_client = mock_rag_client

    service.debug_search("查询", limit=5)

    captured = capsys.readouterr()
    output = captured.out

    assert "未找到任何结果" in output
