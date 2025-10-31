"""Feishu document loader using docx API (correct implementation)"""

import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ..document import Document
from ..config import FeishuConfig
from .base import BaseDataLoader

logger = logging.getLogger(__name__)


class FeishuDocxLoader(BaseDataLoader):
    """飞书文档加载器 - 使用正确的 docx API"""

    FEISHU_API_BASE = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        config: Optional[FeishuConfig] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ):
        """
        初始化飞书文档加载器

        Args:
            config: FeishuConfig 对象
            app_id: 飞书应用 ID
            app_secret: 飞书应用密钥
            document_ids: 要加载的文档 ID 列表
        """
        if config:
            self.config = config
        elif app_id and app_secret:
            self.config = FeishuConfig(app_id=app_id, app_secret=app_secret)
        else:
            # Try to load from environment variables
            try:
                self.config = FeishuConfig.from_env()
            except ValueError as e:
                raise ValueError(
                    "app_id and app_secret are required. "
                    "Set them via config, parameters, or environment variables."
                ) from e

        self.document_ids = document_ids or []
        self.access_token = None
        self.token_expiry = None

    def get_access_token(self) -> str:
        """获取/刷新 Tenant Access Token"""
        # 检查缓存的 token 是否还有效
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token

        try:
            response = requests.post(
                f"{self.FEISHU_API_BASE}/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.config.app_id,
                    "app_secret": self.config.app_secret,
                },
                timeout=10,
            )
            data = response.json()

            if data.get("code") != 0:
                raise Exception(f"Failed to get token: {data.get('msg')}")

            self.access_token = data["tenant_access_token"]
            expire_in = data.get("expire", 7200)
            self.token_expiry = datetime.now() + timedelta(seconds=expire_in - 300)

            logger.info("✅ Feishu Tenant Access Token obtained successfully")
            return self.access_token

        except Exception as e:
            logger.error(f"❌ Failed to authenticate with Feishu: {e}")
            raise

    def get_document_blocks(
        self, document_id: str, page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取文档所有块（支持分页）"""
        token = self.get_access_token()

        try:
            params = {}
            if page_token:
                params["page_token"] = page_token

            response = requests.get(
                f"{self.FEISHU_API_BASE}/docx/v1/documents/{document_id}/blocks",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=10,
            )
            data = response.json()

            if data.get("code") != 0:
                logger.warning(f"Failed to get document blocks {document_id}: {data.get('msg')}")
                return {}

            return data.get("data", {})

        except Exception as e:
            logger.error(f"Failed to get document blocks {document_id}: {e}")
            return {}

    def extract_text_from_element(self, element: Dict) -> str:
        """从元素中提取文本"""
        if "text_run" in element:
            return element["text_run"].get("content", "")
        return ""

    def extract_text_from_blocks(self, blocks: List[Dict]) -> str:
        """从块中提取文本内容"""
        text_parts = []

        for block in blocks:
            block_type = block.get("block_type")

            # block_type 对应关系（来自飞书API文档）
            # 1: page, 2: text, 3: heading1, 4: heading2, 5: heading3
            # 6: heading4, 9: bullet_list, 10: numbered_list, 11: quote
            # 15: code_block, etc.

            if block_type == 2:
                # 文本块
                elements = block.get("text", {}).get("elements", [])
                paragraph_text = ""
                for element in elements:
                    paragraph_text += self.extract_text_from_element(element)
                if paragraph_text.strip():
                    text_parts.append(paragraph_text)

            elif block_type == 3:
                # 一级标题
                elements = block.get("heading1", {}).get("elements", [])
                heading_text = ""
                for element in elements:
                    heading_text += self.extract_text_from_element(element)
                if heading_text.strip():
                    text_parts.append(f"\n# {heading_text}\n")

            elif block_type == 4:
                # 二级标题
                elements = block.get("heading2", {}).get("elements", [])
                heading_text = ""
                for element in elements:
                    heading_text += self.extract_text_from_element(element)
                if heading_text.strip():
                    text_parts.append(f"\n## {heading_text}\n")

            elif block_type == 5:
                # 三级标题
                elements = block.get("heading3", {}).get("elements", [])
                heading_text = ""
                for element in elements:
                    heading_text += self.extract_text_from_element(element)
                if heading_text.strip():
                    text_parts.append(f"\n### {heading_text}\n")

            elif block_type == 6:
                # 四级标题
                elements = block.get("heading4", {}).get("elements", [])
                heading_text = ""
                for element in elements:
                    heading_text += self.extract_text_from_element(element)
                if heading_text.strip():
                    text_parts.append(f"\n#### {heading_text}\n")

            elif block_type == 9:
                # 无序列表
                elements = block.get("bullet", {}).get("elements", [])
                list_text = ""
                for element in elements:
                    list_text += "• " + self.extract_text_from_element(element)
                if list_text.strip():
                    text_parts.append(list_text)

            elif block_type == 10:
                # 有序列表
                elements = block.get("ordered", {}).get("elements", [])
                list_text = ""
                for element in elements:
                    list_text += self.extract_text_from_element(element) + "\n"
                if list_text.strip():
                    text_parts.append(list_text)

            elif block_type == 11:
                # 引用
                elements = block.get("quote", {}).get("elements", [])
                quote_text = ""
                for element in elements:
                    quote_text += self.extract_text_from_element(element)
                if quote_text.strip():
                    text_parts.append(f"> {quote_text}")

            elif block_type == 15:
                # 代码块
                elements = block.get("code_block", {}).get("elements", [])
                code_text = ""
                for element in elements:
                    code_text += self.extract_text_from_element(element)
                if code_text.strip():
                    text_parts.append(f"\n```\n{code_text}\n```\n")

        return "\n".join(text_parts)

    def load_document(self, document_id: str) -> Optional[Document]:
        """加载单个文档"""
        logger.info(f"Loading document: {document_id}")

        all_blocks = []
        page_token = None

        # 分页获取所有块
        while True:
            blocks_data = self.get_document_blocks(document_id, page_token)

            if not blocks_data:
                logger.warning(f"No blocks found for {document_id}")
                break

            blocks = blocks_data.get("items", [])
            all_blocks.extend(blocks)

            # 检查是否有下一页
            if not blocks_data.get("has_more"):
                break

            page_token = blocks_data.get("page_token")

        if not all_blocks:
            logger.warning(f"Empty document: {document_id}")
            return None

        # 提取文本内容
        content = self.extract_text_from_blocks(all_blocks)

        if not content.strip():
            logger.warning(f"No text content extracted from {document_id}")
            return None

        doc = Document(
            content=content,
            doc_id=document_id,
            source="feishu",
            title=f"Feishu Document {document_id[:8]}",
            url=f"https://feishu.cn/docx/{document_id}",
            metadata={
                "doc_type": "docx",
                "block_count": len(all_blocks),
            },
            updated_at=datetime.now(),
        )

        logger.info(f"✅ Loaded document: {document_id} ({len(content)} bytes)")
        return doc

    def load(self) -> List[Document]:
        """加载所有指定的文档"""
        if not self.document_ids:
            logger.warning("No document IDs specified")
            return []

        documents = []

        for doc_id in self.document_ids:
            try:
                doc = self.load_document(doc_id)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load document {doc_id}: {e}")
                continue

        logger.info(f"✅ Successfully loaded {len(documents)} documents from Feishu")
        return self._validate_documents(documents)
