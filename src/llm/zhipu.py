"""Zhipu GLM LLM 客户端实现"""

import logging
from typing import List, Dict
from openai import OpenAI
from .base import BaseLLMClient
from .prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class ZhipuClient(BaseLLMClient):
    """智谱 GLM API 客户端"""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-flash",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        temperature: float = 0.6,
    ):
        """
        初始化 Zhipu 客户端

        Args:
            api_key: Zhipu API 密钥
            model: 模型名称（默认 glm-4-flash）
            base_url: API 基础 URL
            temperature: 采样温度
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.prompt_loader = PromptLoader()

    def generate(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """单轮对话"""
        # 构建上下文
        context_str = "\n\n".join([f"文档片段 {i+1}:\n{doc}" for i, doc in enumerate(context)])

        # 从模板加载 system prompt
        system_prompt = self.prompt_loader.render("system_single.jinja2")

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
            logger.error(f"Zhipu API 调用失败: {e}")
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

        # 从模板加载 system prompt
        system_prompt = self.prompt_loader.render("system_multi.jinja2")

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
            logger.error(f"Zhipu API 调用失败: {e}")
            raise
