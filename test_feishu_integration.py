"""Test script for Feishu integration"""

import logging
from src.rag.data.config import FeishuConfig
from src.rag.data.loaders import FeishuLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def test_feishu_auth():
    """Test Feishu authentication"""
    logger.info("=" * 60)
    logger.info("Testing Feishu Authentication")
    logger.info("=" * 60)

    try:
        config = FeishuConfig.from_env()
        logger.info(f"✅ Config loaded successfully")
        logger.info(f"   App ID: {config.app_id[:10]}***")
        logger.info(f"   Wiki Space ID: {config.wiki_space_id}")

        loader = FeishuLoader(config=config)
        token = loader.get_access_token()
        logger.info(f"✅ Access token obtained: {token[:20]}...")

        return True
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False


def test_wiki_traversal():
    """Test Wiki space traversal"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Wiki Space Traversal")
    logger.info("=" * 60)

    try:
        config = FeishuConfig.from_env()
        loader = FeishuLoader(config=config)

        logger.info(f"Loading documents from Wiki space: {config.wiki_space_id}")

        # Get wiki nodes
        nodes = loader.get_wiki_nodes(config.wiki_space_id)
        logger.info(f"Found {len(nodes)} nodes in wiki space")

        for i, node in enumerate(nodes[:5], 1):  # Show first 5 nodes
            node_type = node.get("type", "unknown")
            title = node.get("title", "Untitled")
            logger.info(f"   {i}. [{node_type}] {title}")

        return True
    except Exception as e:
        logger.error(f"❌ Wiki traversal failed: {e}")
        return False


def test_document_loading():
    """Test document loading"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Document Loading")
    logger.info("=" * 60)

    try:
        config = FeishuConfig.from_env()
        loader = FeishuLoader(config=config)

        logger.info("Loading all documents from Wiki space...")
        documents = loader.load()

        logger.info(f"✅ Successfully loaded {len(documents)} documents")

        if documents:
            logger.info("\nFirst 3 documents:")
            for i, doc in enumerate(documents[:3], 1):
                content_preview = doc.content[:100].replace("\n", " ")
                logger.info(
                    f"   {i}. {doc.title}")
                logger.info(f"      ID: {doc.doc_id}")
                logger.info(f"      URL: {doc.url}")
                logger.info(f"      Content: {content_preview}...")

        return True
    except Exception as e:
        logger.error(f"❌ Document loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n🚀 Starting Feishu Integration Tests\n")

    results = {
        "Authentication": test_feishu_auth(),
        "Wiki Traversal": test_wiki_traversal(),
        "Document Loading": test_document_loading(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    logger.info("=" * 60)

    if all_passed:
        logger.info("🎉 All tests passed!")
    else:
        logger.info("⚠️ Some tests failed. Check the logs above.")

    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
