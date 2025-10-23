"""Test using document token directly"""

import requests
import logging
from src.rag.data.config import FeishuConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_document_token():
    """Test getting document content and metadata using token"""

    config = FeishuConfig.from_env()

    # Get access token
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={"app_id": config.app_id, "app_secret": config.app_secret}
    )
    token = response.json()["app_access_token"]
    logger.info(f"✅ Got access token")

    doc_token = "GZctwJerPiiqUgkGRb4cR3vhnhf"

    # Try 1: Get document metadata
    logger.info(f"\n{'='*60}")
    logger.info("Getting document metadata...")
    logger.info(f"{'='*60}")

    response = requests.get(
        f"https://open.feishu.cn/open-apis/docs/v2/{doc_token}",
        headers={"Authorization": f"Bearer {token}"}
    )
    logger.info(f"Status: {response.status_code}")
    meta_data = response.json()
    logger.info(f"Metadata: {meta_data}\n")

    # Try 2: Get document content
    logger.info(f"{'='*60}")
    logger.info("Getting document content...")
    logger.info(f"{'='*60}")

    response = requests.get(
        f"https://open.feishu.cn/open-apis/docs/v2/{doc_token}/raw_content",
        headers={"Authorization": f"Bearer {token}"}
    )
    logger.info(f"Status: {response.status_code}")
    content_data = response.json()

    if content_data.get("code") == 0:
        content = content_data.get("data", {}).get("content", "")
        logger.info(f"✅ Got document content ({len(content)} bytes)")
        logger.info(f"\nFirst 500 characters:\n{content[:500]}\n")
    else:
        logger.error(f"Error: {content_data}")

    # Try 3: Get children/outline (if this is a folder-like document)
    logger.info(f"{'='*60}")
    logger.info("Getting document outline/blocks...")
    logger.info(f"{'='*60}")

    response = requests.get(
        f"https://open.feishu.cn/open-apis/docs/v2/{doc_token}/blocks",
        headers={"Authorization": f"Bearer {token}"}
    )
    logger.info(f"Status: {response.status_code}")
    blocks_data = response.json()
    logger.info(f"Blocks: {blocks_data}\n")


if __name__ == "__main__":
    test_document_token()
