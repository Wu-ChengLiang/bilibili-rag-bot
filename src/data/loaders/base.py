"""Base data loader class"""

from abc import ABC, abstractmethod
from typing import List
from ..document import Document


class BaseDataLoader(ABC):
    """所有数据加载器的基类"""

    @abstractmethod
    def load(self) -> List[Document]:
        """加载数据，返回 Document 列表"""
        raise NotImplementedError

    def _validate_documents(self, docs: List[Document]) -> List[Document]:
        """验证文档有效性"""
        return [d for d in docs if d.content and d.content.strip()]
