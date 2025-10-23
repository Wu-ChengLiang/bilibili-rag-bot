"""Debug script to see actual block structure"""

import requests
import json
import logging
from src.rag.data.config import FeishuConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = FeishuConfig.from_env()

    # Get token
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": config.app_id, "app_secret": config.app_secret},
    )
    token = response.json()["tenant_access_token"]
    logger.info(f"✅ Got token")

    # Get blocks
    document_id = "XCJzwF6Pqi1t5UkUVnpcCSsQnQd"
    response = requests.get(
        f"https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}/blocks",
        headers={"Authorization": f"Bearer {token}"},
    )

    data = response.json()
    logger.info(f"\n{'='*60}")
    logger.info("Full API Response:")
    logger.info(f"{'='*60}")
    logger.info(json.dumps(data, indent=2, ensure_ascii=False))

    # Print just the blocks structure
    if data.get("code") == 0:
        blocks = data.get("data", {}).get("items", [])
        logger.info(f"\n{'='*60}")
        logger.info(f"Block Types Found: {len(blocks)}")
        logger.info(f"{'='*60}")

        for i, block in enumerate(blocks[:5]):  # Show first 5 blocks
            logger.info(f"\nBlock {i + 1}:")
            logger.info(f"  Type: {block.get('type')}")
            logger.info(f"  Block ID: {block.get('block_id')}")
            logger.info(f"  Keys: {list(block.keys())}")
            logger.info(f"  Full: {json.dumps(block, indent=4, ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    main()
