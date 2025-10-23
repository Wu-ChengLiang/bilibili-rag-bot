"""Test Feishu docx loader"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from src.data.config import FeishuConfig
from src.data.loaders import FeishuDocxLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Testing Feishu Docx Loader")
    logger.info("=" * 60)

    # 创建配置
    config = FeishuConfig.from_env()

    # 创建加载器，指定要加载的文档 ID
    # 使用你提供的 Wiki 文档 ID
    document_id = "XCJzwF6Pqi1t5UkUVnpcCSsQnQd"

    loader = FeishuDocxLoader(config=config, document_ids=[document_id])

    logger.info(f"\nLoading document: {document_id}\n")

    try:
        documents = loader.load()

        if documents:
            logger.info(f"✅ Successfully loaded {len(documents)} document(s)")
            logger.info("\n" + "=" * 60)
            logger.info("Document Details")
            logger.info("=" * 60)

            for i, doc in enumerate(documents, 1):
                logger.info(f"\nDocument {i}:")
                logger.info(f"  ID: {doc.doc_id}")
                logger.info(f"  Title: {doc.title}")
                logger.info(f"  Source: {doc.source}")
                logger.info(f"  URL: {doc.url}")
                logger.info(f"  Content Length: {len(doc.content)} bytes")
                logger.info(f"  Content Preview: {doc.content[:200]}...")
                logger.info(f"  Metadata: {doc.metadata}")
        else:
            logger.warning("❌ No documents loaded")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
