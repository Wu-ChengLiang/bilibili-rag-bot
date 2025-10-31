"""Kimi/Moonshot LLM 客户端实现"""

import logging
from typing import List, Dict
from openai import OpenAI
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class KimiClient(BaseLLMClient):
    """Kimi/Moonshot API 客户端"""

    def __init__(
        self,
        api_key: str,
        model: str = "moonshot-v1-8k",
        base_url: str = "https://api.moonshot.cn/v1",
        temperature: float = 0.6,
    ):
        """
        初始化 Kimi 客户端

        Args:
            api_key: Moonshot API 密钥
            model: 模型名称
            base_url: API 基础 URL
            temperature: 采样温度
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """单轮对话"""
        # 构建上下文
        context_str = "\n\n".join([f"文档片段 {i+1}:\n{doc}" for i, doc in enumerate(context)])

        system_prompt = """你是一个智能问答助手。你会基于提供的文档片段来回答用户的问题。

请仔细阅读文档片段，从中找到最相关的信息来回答问题。如果文档中没有相关信息，请说明这一点。"""

        user_message = f"""参考文档：
{context_str}

用户问题：{query}

请基于上述文档回答用户的问题。"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Kimi API 调用失败: {e}")
            raise

    def generate_with_history(
        self,
        query: str,
        context: List[str],
        history: List[Dict[str, str]],
    ) -> str:
        """多轮对话 - 考虑对话历史"""
        # 构建上下文
        context_str = "\n\n".join([f"文档片段 {i+1}:\n{doc}" for i, doc in enumerate(context)])

        system_prompt = """你是一个智能问答助手。你会基于提供的文档片段和之前的对话历史来回答用户的问题。

请仔细阅读文档片段和对话历史，理解用户的意思和上下文，然后给出准确的回答。
如果文档中没有相关信息，请说明这一点。"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # 添加历史消息
        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        # 添加当前消息和上下文
        user_message = f"""参考文档：
{context_str}

用户的当前问题：{query}

请基于上述文档和之前的对话历史回答用户的问题。"""

        messages.append({
            "role": "user",
            "content": user_message,
        })

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Kimi API 调用失败: {e}")
            raise
