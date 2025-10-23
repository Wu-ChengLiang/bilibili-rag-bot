"""Test script to get numeric wiki space ID"""

import requests
import logging
from src.rag.data.config import FeishuConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_user_wiki_spaces():
    """Get all wiki spaces accessible to the user"""
    try:
        config = FeishuConfig.from_env()

        # Get access token
        response = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
            json={"app_id": config.app_id, "app_secret": config.app_secret}
        )
        token = response.json()["app_access_token"]

        # Get wiki spaces
        response = requests.get(
            "https://open.feishu.cn/open-apis/wiki/v2/spaces",
            headers={"Authorization": f"Bearer {token}"}
        )

        data = response.json()
        print(f"\n✅ Response: {data}\n")

        if data.get("code") == 0:
            spaces = data.get("data", {}).get("items", [])
            logger.info(f"Found {len(spaces)} wiki spaces:")
            for space in spaces:
                logger.info(f"  - Name: {space.get('name')}")
                logger.info(f"    Space ID (numeric): {space.get('space_id')}")
                logger.info(f"    Space Token: {space.get('space_id')}")
                logger.info("")
        else:
            logger.error(f"Error: {data.get('msg')}")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    get_user_wiki_spaces()
