"""Feishu document loader with Wiki space traversal"""

import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ..document import Document
from ..config import FeishuConfig
from .base import BaseDataLoader

logger = logging.getLogger(__name__)


class FeishuLoader(BaseDataLoader):
    """飞书文档加载器 - 支持Wiki空间自动遍历"""

    FEISHU_API_BASE = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        config: Optional[FeishuConfig] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        wiki_space_id: Optional[str] = None,
    ):
        """
        初始化飞书加载器

        Args:
            config: FeishuConfig 对象
            app_id: 飞书应用 ID (如果未提供 config)
            app_secret: 飞书应用密钥 (如果未提供 config)
            wiki_space_id: Wiki 空间 ID (如果未提供 config)
        """
        if config:
            self.config = config
        else:
            self.config = FeishuConfig(
                app_id=app_id or "",
                app_secret=app_secret or "",
                wiki_space_id=wiki_space_id,
            )

        if not self.config.app_id or not self.config.app_secret:
            raise ValueError(
                "app_id and app_secret are required. "
                "Set them via config or environment variables."
            )

        self.access_token = None
        self.token_expiry = None

    def get_access_token(self) -> str:
        """获取/刷新访问令牌"""
        # 检查缓存的 token 是否还有效
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token

        try:
            response = requests.post(
                f"{self.FEISHU_API_BASE}/auth/v3/app_access_token/internal",
                json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
                timeout=10,
            )
            data = response.json()

            if data.get("code") != 0:
                raise Exception(f"Failed to get token: {data.get('msg')}")

            self.access_token = data["app_access_token"]
            expire_in = data.get("expire", 7200)
            self.token_expiry = datetime.now() + timedelta(seconds=expire_in - 300)

            logger.info("✅ Feishu access token obtained successfully")
            return self.access_token

        except Exception as e:
            logger.error(f"❌ Failed to authenticate with Feishu: {e}")
            raise

    def get_wiki_nodes(self, space_id: str) -> List[Dict[str, Any]]:
        """获取Wiki空间中的所有节点（文件夹和文档）"""
        token = self.get_access_token()
        nodes = []

        try:
            response = requests.get(
                f"{self.FEISHU_API_BASE}/wiki/v2/spaces/{space_id}/nodes",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            data = response.json()

            if data.get("code") != 0:
                logger.warning(f"Failed to get wiki nodes: {data.get('msg')}")
                return []

            nodes = data.get("data", {}).get("items", [])
            logger.info(f"Found {len(nodes)} nodes in wiki space")
            return nodes

        except Exception as e:
            logger.error(f"Failed to get wiki nodes: {e}")
            return []

    def get_document_content(self, doc_id: str) -> str:
        """获取飞书文档的 markdown 内容"""
        token = self.get_access_token()

        try:
            response = requests.get(
                f"{self.FEISHU_API_BASE}/docs/v2/{doc_id}/raw_content",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            data = response.json()

            if data.get("code") != 0:
                logger.warning(f"Failed to get document {doc_id}: {data.get('msg')}")
                return ""

            return data.get("data", {}).get("content", "")

        except Exception as e:
            logger.error(f"Failed to get document content {doc_id}: {e}")
            return ""

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """获取文档元数据"""
        token = self.get_access_token()

        try:
            response = requests.get(
                f"{self.FEISHU_API_BASE}/docs/v2/{doc_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            data = response.json()

            if data.get("code") != 0:
                logger.warning(f"Failed to get metadata for {doc_id}")
                return {}

            return data.get("data", {})

        except Exception as e:
            logger.error(f"Failed to get document metadata {doc_id}: {e}")
            return {}

    def traverse_wiki(self, space_id: str) -> List[Document]:
        """递归遍历Wiki空间，获取所有文档"""
        documents = []
        nodes = self.get_wiki_nodes(space_id)

        for node in nodes:
            try:
                node_type = node.get("type")
                node_id = node.get("node_token")
                title = node.get("title", "")

                if node_type == "doc":
                    # 这是一个文档
                    logger.info(f"Loading document: {title}")

                    content = self.get_document_content(node_id)
                    if not content:
                        logger.warning(f"Empty content for document {title}")
                        continue

                    meta = self.get_document_metadata(node_id)

                    doc = Document(
                        content=content,
                        doc_id=node_id,
                        source="feishu",
                        title=title,
                        url=f"https://feishu.cn/docs/{node_id}",
                        metadata={
                            "owner": meta.get("owner_id", ""),
                            "created_at": meta.get("create_time"),
                            "updated_at": meta.get("update_time"),
                            "wiki_space_id": space_id,
                            "node_type": "document",
                        },
                        updated_at=datetime.now(),
                    )

                    documents.append(doc)
                    logger.info(f"✅ Loaded document: {title}")

                elif node_type == "folder":
                    # 这是一个文件夹，递归进去
                    logger.info(f"Traversing folder: {title}")
                    sub_docs = self.traverse_wiki(space_id)
                    documents.extend(sub_docs)

            except Exception as e:
                logger.error(f"Error processing node {node.get('title')}: {e}")
                continue

        return documents

    def load(self) -> List[Document]:
        """加载飞书Wiki空间中的所有文档"""
        if not self.config.wiki_space_id:
            raise ValueError(
                "wiki_space_id is required. "
                "Set FEISHU_WIKI_SPACE_ID in .env file or pass it as parameter."
            )

        logger.info(f"Starting to load documents from Wiki space: {self.config.wiki_space_id}")

        try:
            documents = self.traverse_wiki(self.config.wiki_space_id)
            documents = self._validate_documents(documents)

            logger.info(f"✅ Successfully loaded {len(documents)} documents from Feishu")
            return documents

        except Exception as e:
            logger.error(f"❌ Failed to load documents from Feishu: {e}")
            raise
