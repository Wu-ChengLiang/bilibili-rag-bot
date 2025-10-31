"""Jinja2 Template-based Prompt Loader

提供统一的 prompt 加载和渲染接口，支持所有 LLM 实现
"""

from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
from pathlib import Path


class PromptLoader:
    """Jinja2 模板加载器 - 负责加载和渲染 prompt 模板"""

    def __init__(self, template_dir: str = None):
        """
        初始化 Prompt 加载器

        Args:
            template_dir: 模板目录路径，默认为 src/llm/prompts
        """
        if template_dir is None:
            # 默认路径：当前文件同级目录的 prompts 文件夹
            template_dir = os.path.join(os.path.dirname(__file__), "prompts")

        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(enabled_extensions=("jinja2",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **kwargs) -> str:
        """
        加载并渲染模板

        Args:
            template_name: 模板文件名（如 'system_single.jinja2'）
            **kwargs: 传递给模板的变量

        Returns:
            渲染后的模板内容

        Raises:
            FileNotFoundError: 模板文件不存在
            Exception: 模板渲染错误
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to render template '{template_name}': {str(e)}")

    def list_templates(self) -> list:
        """列出所有可用的模板"""
        if not os.path.exists(self.template_dir):
            return []
        return [f for f in os.listdir(self.template_dir) if f.endswith(".jinja2")]
