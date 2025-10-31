"""API 请求和响应模型"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class MessageHistory(BaseModel):
    """对话历史中的单条消息"""
    role: str = Field(..., description="角色: 'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    platform: str = Field(..., description="平台名称，如: bilibili, weibo")
    user_id: str = Field(..., description="用户 ID")
    user_name: str = Field(..., description="用户名")
    message: str = Field(..., description="用户消息")
    history: Optional[List[MessageHistory]] = Field(
        default=None,
        description="对话历史（可选）",
    )

    class Config:
        example = {
            "platform": "bilibili",
            "user_id": "123456",
            "user_name": "阿良",
            "message": "什么是向量数据库？",
            "history": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么我可以帮助的吗？"},
            ],
        }


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool = Field(..., description="是否成功")
    reply: Optional[str] = Field(default=None, description="回复内容")
    error: Optional[str] = Field(default=None, description="错误信息")

    class Config:
        example = {
            "success": True,
            "reply": "向量数据库是存储和查询高维向量的数据库系统...",
        }
