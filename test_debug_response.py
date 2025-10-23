"""Debug API responses"""

import requests
import logging
from src.rag.data.config import FeishuConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_responses():
    """Test and print raw responses"""

    config = FeishuConfig.from_env()

    # Get access token
    response = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={"app_id": config.app_id, "app_secret": config.app_secret}
    )
    token = response.json()["app_access_token"]

    doc_token = "GZctwJerPiiqUgkGRb4cR3vhnhf"

    # Test metadata endpoint
    logger.info("Testing metadata endpoint...")
    response = requests.get(
        f"https://open.feishu.cn/open-apis/docs/v2/{doc_token}",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Raw text: {response.text[:500]}")
    print(f"Raw content: {response.content[:500]}")


if __name__ == "__main__":
    test_responses()
