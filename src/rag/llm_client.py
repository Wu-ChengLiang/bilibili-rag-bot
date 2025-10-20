"""LLM Client for RAG system - Kimi API integration"""

from typing import List, Dict, Any, Optional
from openai import OpenAI


class LLMClient:
    """LLM Client for generating responses with retrieved context"""

    def __init__(
        self,
        api_key: str,
        model: str = "moonshot-v1-8k",
        base_url: str = "https://api.moonshot.cn/v1",
        temperature: float = 0.6
    ):
        """Initialize LLM client

        Args:
            api_key: Kimi API key
            model: Model name
            base_url: API base URL
            temperature: Sampling temperature
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using retrieved context

        Args:
            query: User query
            context: Retrieved document chunks
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        # Build context string
        context_str = "\n\n".join([
            f"文档片段 {i+1}:\n{doc}"
            for i, doc in enumerate(context)
        ])

        # Default system prompt
        if system_prompt is None:
            system_prompt = """你是一个智能问答助手。你会基于提供的文档片段来回答用户的问题。

请仔细阅读文档片段，从中找到最相关的信息来回答问题。"""

        # Build user message with context
        user_message = f"""参考文档：
{context_str}

用户问题：{query}

请基于上述文档回答用户的问题。"""

        # Call API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature
        )

        return completion.choices[0].message.content

    def generate_simple(self, query: str) -> str:
        """Generate response without context (simple chat)

        Args:
            query: User query

        Returns:
            Generated response
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手。"},
                {"role": "user", "content": query}
            ],
            temperature=self.temperature
        )

        return completion.choices[0].message.content
