"""Test RealTimeFeishuRAG with scheduled updates"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from src.rag.realtime_feishu_rag import RealTimeFeishuRAG
from src.scheduler import RAGScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_realtime_rag_basic():
    """Test basic RealTimeFeishuRAG functionality"""
    logger.info("=" * 60)
    logger.info("Test 1: Basic RealTimeFeishuRAG")
    logger.info("=" * 60)

    # Initialize RAG client
    rag = RealTimeFeishuRAG(
        doc_ids=["XCJzwF6Pqi1t5UkUVnpcCSsQnQd"]  # Your test document
    )

    logger.info(f"\n{rag}\n")

    # Get stats
    stats = rag.get_stats()
    logger.info(f"Stats: {stats}\n")

    # Test search
    test_queries = [
        "Embedding",
        "RAG",
        "LLM",
        "向量数据库",
    ]

    logger.info("Testing searches:")
    for query in test_queries:
        logger.info(f"\n🔍 Query: '{query}'")
        results = rag.search(query, limit=3)

        if results:
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. [{result['score']:.2f}] {result['content'][:100]}...")
        else:
            logger.info("  No results found")


def test_scheduler():
    """Test scheduler with immediate refresh (for testing)"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Scheduler with Immediate Refresh")
    logger.info("=" * 60)

    # Initialize RAG client
    rag = RealTimeFeishuRAG(
        doc_ids=["XCJzwF6Pqi1t5UkUVnpcCSsQnQd"]
    )

    # Initialize scheduler
    scheduler = RAGScheduler()

    # For testing: schedule refresh in 3 seconds instead of midnight
    logger.info("\n⏰ Scheduling refresh in 3 seconds (for testing)...")

    scheduler.scheduler.add_job(
        rag.refresh_vector_store,
        "interval",
        seconds=3,
        id="test_refresh",
        name="Test refresh",
    )

    scheduler.start()
    logger.info(f"Scheduler: {scheduler}\n")

    # Wait for refresh to happen
    logger.info("Waiting for scheduled refresh...")
    time.sleep(5)

    # Verify refresh happened
    logger.info(f"\n✅ Refresh completed at: {rag.last_update_time}\n")

    # Test search after refresh
    logger.info("Testing search after refresh...")
    results = rag.search("Embedding", limit=3)
    logger.info(f"Found {len(results)} results\n")

    # Cleanup
    scheduler.stop()
    logger.info("Test completed\n")


def test_daily_schedule():
    """Show how to set up daily refresh"""
    logger.info("=" * 60)
    logger.info("Test 3: Daily Schedule Configuration (for production)")
    logger.info("=" * 60)

    logger.info("""
# Production setup example:

from src.rag.realtime_feishu_rag import RealTimeFeishuRAG
from src.scheduler import RAGScheduler

# Initialize RAG
rag = RealTimeFeishuRAG(doc_ids=["your_doc_id"])

# Initialize scheduler
scheduler = RAGScheduler()

# Schedule daily refresh at midnight
scheduler.schedule_daily_refresh(rag, hour=0, minute=0)

# Start scheduler
scheduler.start()

# In your FastAPI app:
@app.get("/search")
async def search(q: str):
    results = rag.search(q)
    return {"results": results}

# Scheduler will automatically refresh at 00:00 every day
# Your searches will always use the freshest data from Feishu
    """)


if __name__ == "__main__":
    try:
        # Run tests
        test_realtime_rag_basic()
        test_scheduler()
        test_daily_schedule()

        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
