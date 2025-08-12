"""
Celery application setup for background processing.

This module defines the Celery app configured to use Redis as both the
broker and result backend. Tasks are defined in follow-up edits.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging

# Ensure logging is configured for workers
setup_logging()
logger = get_logger(__name__)


def _create_celery_app() -> Celery:
    """Create and configure the Celery app instance."""
    broker_url = settings.REDIS_URL
    result_backend = settings.REDIS_URL

    app = Celery("chatbot_background_jobs", broker=broker_url, backend=result_backend)

    # Sensible defaults for reliable processing
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        worker_prefetch_multiplier=1,  # prefer fairness with retries
        task_acks_late=True,  # don't lose tasks on worker crash
        broker_transport_options={
            "visibility_timeout": 3600,  # 1 hour
        },
        result_expires=3600,  # 1 hour
    )

    logger.info("Celery app initialized", extra={"backend": "redis"})
    return app


# Celery application singleton used by worker and beat
celery_app: Celery = _create_celery_app()


@celery_app.task(bind=True, max_retries=3, name="jobs.process_document_async")
def process_document_async(self, job_id: str, document_data: dict) -> dict:
    """Background document processing task (stub)."""
    logger.info(
        "process_document_async invoked",
        extra={"job_id": job_id, "payload_keys": list(document_data.keys())},
    )
    # TODO: integrate with IngestionEngine in subsequent sprint
    return {"job_id": job_id, "status": "queued"}


@celery_app.task(bind=True, max_retries=5, retry_backoff=True, name="jobs.scrape_url_async")
def scrape_url_async(self, job_id: str, url: str, options: dict | None = None) -> dict:
    """Background URL scraping task (stub)."""
    logger.info(
        "scrape_url_async invoked", extra={"job_id": job_id, "url": url, "options": options or {}}
    )
    return {"job_id": job_id, "status": "queued"}


@celery_app.task(name="jobs.cleanup_failed_jobs")
def cleanup_failed_jobs() -> int:
    """Periodic cleanup task (stub). Returns number of cleaned items."""
    logger.info("cleanup_failed_jobs invoked")
    # Placeholder count until real job store is implemented
    return 0
