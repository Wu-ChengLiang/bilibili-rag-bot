"""LLM 抽象基类 - 定义所有 LLM 实现的接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMClient(ABC):
    """所有 LLM 客户端的抽象基类"""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """
        单轮对话 - 不考虑历史

        Args:
            query: 用户问题
            context: 检索到的相关文档片段

        Returns:
            LLM 生成的回答
        """
        raise NotImplementedError

    @abstractmethod
    def generate_with_history(
        self,
        query: str,
        context: List[str],
        history: List[Dict[str, str]],
    ) -> str:
        """
        多轮对话 - 考虑对话历史

        Args:
            query: 当前用户问题
            context: 检索到的相关文档片段
            history: 对话历史，格式：[{"role": "user"/"assistant", "content": "..."}, ...]

        Returns:
            LLM 生成的回答
        """
        raise NotImplementedError
