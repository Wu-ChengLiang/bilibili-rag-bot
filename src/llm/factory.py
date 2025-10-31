"""LLM 工厂 - 根据提供者创建 LLM 客户端"""

from typing import Dict, Type
from .base import BaseLLMClient
from .kimi import KimiClient

# 提供者注册表
_providers: Dict[str, Type[BaseLLMClient]] = {
    "kimi": KimiClient,
    "moonshot": KimiClient,  # moonshot 是 kimi 的别名
}


class LLMFactory:
    """LLM 工厂类 - 工厂模式实现"""

    @staticmethod
    def create(provider: str, api_key: str, **kwargs) -> BaseLLMClient:
        """
        创建 LLM 客户端

        Args:
            provider: 提供者名称 ("kimi", "moonshot", 等)
            api_key: API 密钥
            **kwargs: 其他参数（model, temperature 等）

        Returns:
            LLM 客户端实例

        Raises:
            ValueError: 未知的提供者
        """
        if provider not in _providers:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Available providers: {list(_providers.keys())}"
            )

        client_class = _providers[provider]
        return client_class(api_key=api_key, **kwargs)

    @staticmethod
    def register(provider: str, client_class: Type[BaseLLMClient]) -> None:
        """
        注册新的 LLM 提供者

        Args:
            provider: 提供者名称
            client_class: 客户端类（必须继承 BaseLLMClient）
        """
        if not issubclass(client_class, BaseLLMClient):
            raise TypeError(
                f"{client_class} must inherit from BaseLLMClient"
            )
        _providers[provider] = client_class

    @staticmethod
    def list_providers() -> list:
        """列出所有可用的提供者"""
        return list(_providers.keys())
