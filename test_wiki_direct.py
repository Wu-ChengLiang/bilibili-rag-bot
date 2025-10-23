"""Test direct wiki node retrieval with different approaches"""

import requests
import logging
from src.rag.data.config import FeishuConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_wiki_nodes_with_different_formats():
    """Try different wiki space ID formats"""

    config = FeishuConfig.from_env()

    # Get access token
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={"app_id": config.app_id, "app_secret": config.app_secret}
    )
    token = response.json()["app_access_token"]

    # Try different space ID formats
    space_ids_to_try = [
        config.wiki_space_id,  # Original: GZctwJerPiiqUgkGRb4cR3vhnhf
    ]

    for space_id in space_ids_to_try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Trying space_id: {space_id}")
        logger.info(f"{'='*60}")

        # Try 1: Using /wiki/v2/spaces/{space_id}/nodes
        logger.info("Method 1: /wiki/v2/spaces/{space_id}/nodes")
        response = requests.get(
            f"https://open.feishu.cn/open-apis/wiki/v2/spaces/{space_id}/nodes",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")

        # Try 2: Using /drive/v1/files with folder_token
        logger.info("Method 2: /drive/v1/files with space as folder_token")
        response = requests.get(
            f"https://open.feishu.cn/open-apis/drive/v1/files/{space_id}/children",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")

        # Try 3: List all files to see structure
        logger.info("Method 3: /drive/v1/files with query")
        response = requests.get(
            "https://open.feishu.cn/open-apis/drive/v1/files",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "parent_token": space_id,
                "page_size": 50
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")


if __name__ == "__main__":
    test_wiki_nodes_with_different_formats()
