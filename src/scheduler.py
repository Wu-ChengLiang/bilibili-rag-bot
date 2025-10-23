"""Scheduled tasks for RAG system"""

import logging
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class RAGScheduler:
    """
    Manage scheduled tasks for RAG system.

    Currently supports:
    - Daily vector store refresh at midnight (00:00)
    """

    def __init__(self):
        """Initialize scheduler"""
        self.scheduler = BackgroundScheduler()
        self.is_running = False

    def schedule_daily_refresh(
        self,
        rag_client,
        hour: int = 0,
        minute: int = 0,
    ) -> None:
        """
        Schedule daily vector store refresh.

        Args:
            rag_client: RealTimeFeishuRAG instance
            hour: Hour to refresh (0-23, default 0 = midnight)
            minute: Minute to refresh (0-59, default 0)
        """
        logger.info(f"Scheduling daily refresh at {hour:02d}:{minute:02d}")

        self.scheduler.add_job(
            rag_client.refresh_vector_store,
            CronTrigger(hour=hour, minute=minute),
            id="daily_refresh",
            name="Daily vector store refresh",
            misfire_grace_time=60,  # Allow 1 min grace
            coalesce=True,  # Run once if multiple missed
            max_instances=1,  # Only one instance at a time
        )

    def start(self) -> None:
        """Start the scheduler"""
        if not self.is_running:
            logger.info("🚀 Starting scheduler...")
            self.scheduler.start()
            self.is_running = True
            logger.info("✅ Scheduler started")
        else:
            logger.warning("Scheduler is already running")

    def stop(self) -> None:
        """Stop the scheduler"""
        if self.is_running:
            logger.info("⏹️ Stopping scheduler...")
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("✅ Scheduler stopped")
        else:
            logger.warning("Scheduler is not running")

    def get_jobs(self):
        """Get all scheduled jobs"""
        return self.scheduler.get_jobs()

    def __repr__(self) -> str:
        return f"RAGScheduler(running={self.is_running}, jobs={len(self.get_jobs())})"
