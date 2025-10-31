"""对话历史管理 - 按平台分类存储用户的对话历史"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """对话历史管理器 - 维护用户的对话历史"""

    def __init__(self, base_dir: str = "./history"):
        """
        初始化历史管理器

        Args:
            base_dir: 历史文件存储的根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

    def _get_history_path(self, platform: str, user_id: str) -> Path:
        """获取用户的历史文件路径"""
        # 按平台分文件夹: history/bilibili/123456.json
        platform_dir = self.base_dir / platform
        platform_dir.mkdir(exist_ok=True, parents=True)
        return platform_dir / f"{user_id}.json"

    def load_history(
        self,
        platform: str,
        user_id: str,
    ) -> List[Dict[str, str]]:
        """
        加载用户的对话历史

        Args:
            platform: 平台名称 (bilibili, weibo, etc.)
            user_id: 用户 ID

        Returns:
            对话历史列表，格式: [{"role": "user"/"assistant", "content": "..."}, ...]
        """
        history_path = self._get_history_path(platform, user_id)

        if not history_path.exists():
            logger.info(f"No history found for {platform}/{user_id}")
            return []

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} messages from {platform}/{user_id}")
                return data
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []

    def save_history(
        self,
        platform: str,
        user_id: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        保存用户的对话历史

        Args:
            platform: 平台名称
            user_id: 用户 ID
            history: 对话历史列表
        """
        history_path = self._get_history_path(platform, user_id)

        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(history)} messages to {platform}/{user_id}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            raise

    def add_message(
        self,
        platform: str,
        user_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        添加一条消息到历史

        Args:
            platform: 平台名称
            user_id: 用户 ID
            role: 角色 ("user" 或 "assistant")
            content: 消息内容
        """
        # 加载现有历史
        history = self.load_history(platform, user_id)

        # 添加新消息
        history.append({
            "role": role,
            "content": content,
        })

        # 保存更新后的历史
        self.save_history(platform, user_id, history)

    def clear_history(self, platform: str, user_id: str) -> None:
        """清空用户的对话历史"""
        history_path = self._get_history_path(platform, user_id)
        try:
            if history_path.exists():
                history_path.unlink()
                logger.info(f"Cleared history for {platform}/{user_id}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise

    def get_latest_messages(
        self,
        platform: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, str]]:
        """获取最近 N 条消息（用于限制 LLM 输入长度）"""
        history = self.load_history(platform, user_id)
        return history[-limit:] if len(history) > limit else history
