"""Simple test to get document content directly"""

import requests
import logging
from src.rag.data.config import FeishuConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_simple():
    """Simple direct test"""

    config = FeishuConfig.from_env()

    # Get access token
    logger.info("Getting access token...")
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={"app_id": config.app_id, "app_secret": config.app_secret}
    )
    data = response.json()

    if data.get("code") != 0:
        logger.error(f"Failed to get token: {data}")
        return

    token = data["app_access_token"]
    logger.info(f"✅ Got token: {token[:20]}...")

    # Try to get raw content
    doc_token = config.wiki_space_id
    logger.info(f"\nTrying to get raw content for: {doc_token}")

    response = requests.get(
        f"https://open.feishu.cn/open-apis/docs/v2/{doc_token}/raw_content",
        headers={"Authorization": f"Bearer {token}"}
    )

    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Response Headers: {dict(response.headers)}")
    logger.info(f"Response Text: {response.text[:500]}")

    # Try parsing as JSON
    try:
        json_data = response.json()
        logger.info(f"\n✅ JSON Response:")
        logger.info(f"{json_data}")

        if json_data.get("code") == 0:
            content = json_data.get("data", {}).get("content", "")
            logger.info(f"\n✅✅ Successfully got content!")
            logger.info(f"Content length: {len(content)} bytes")
            logger.info(f"First 300 chars:\n{content[:300]}")
        else:
            logger.error(f"API Error: {json_data.get('msg')}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")


if __name__ == "__main__":
    test_simple()
