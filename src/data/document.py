"""Standard Document class for all data sources"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class Document:
    """标准文档对象 - 所有数据源的统一格式"""

    content: str                    # 文档内容（用于 embedding）
    doc_id: str                     # 唯一 ID
    source: str                     # 数据源: "feishu", "local", "web" 等

    # 可选元数据
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 时间戳
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def summary(self) -> str:
        """获取文档摘要（前100字）"""
        return self.content[:100] + "..." if len(self.content) > 100 else self.content

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
