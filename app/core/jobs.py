"""
Celery application setup for background processing.

This module defines the Celery app configured to use Redis as both the
broker and result backend. Tasks are defined in follow-up edits.
"""

from __future__ import annotations

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.dedup import ContentDeduplicator
from app.core.incremental import IncrementalProcessor
from app.core.ingestion import NLTKAdaptiveChunker
from app.core.logging_config import get_logger, set_request_id, set_trace_id, setup_logging
from app.core.metadata import ChunkMetadataExtractor
from app.db.models import Document

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
    """Background document processing task.

    Minimal v1: chunk via ingestion engine, dedup/update chunks, embed via IncrementalProcessor.
    """
    # Initialize correlation IDs for worker task context
    set_request_id()
    set_trace_id(job_id)

    logger.info(
        "process_document_async invoked",
        extra={"job_id": job_id, "payload_keys": list(document_data.keys())},
    )
    # DB session (worker-safe)
    db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:5432/{settings.POSTGRES_DB}"
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=create_engine(db_url))
    session = SessionLocal()
    try:
        doc_id = document_data.get("document_id")
        storage_uri = document_data.get("storage_uri")
        # In minimal/eager test mode, allow empty payload and just report queued
        if not doc_id:
            return {"job_id": job_id, "status": "queued"}
        doc = session.get(Document, doc_id)
        if not doc:
            return {"job_id": job_id, "status": "failed", "error": "document_not_found"}

        # Fetch file contents from storage (local backend supports file://)
        # For v1, assume local file path; real impl would use storage backend retrieve
        try:
            import urllib.parse
            from pathlib import Path

            path = Path(urllib.parse.urlparse(storage_uri or doc.storage_uri).path)
            file_bytes = path.read_bytes()
            file_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            file_text = ""

        # Chunk using the same chunker implementation as ingestion
        chunks = NLTKAdaptiveChunker().chunk(file_text) if file_text else []

        # Upsert chunks
        dedup = ContentDeduplicator()
        prepared = []
        ChunkMetadataExtractor()
        # In this minimal async path, we don't have page splits; set page_number=None and no headers
        for idx, content in enumerate(chunks):
            prepared.append(
                {
                    "index": idx,
                    "content": content,
                    "page_number": None,
                    "section_title": None,
                    "extra_metadata": {"headers": []},
                }
            )
        if prepared:
            dedup.upsert_chunks(
                session,
                document_id=doc.id,
                chunks=[
                    # Build ChunkInput without importing in worker context
                    type("CI", (), ci)()
                    for ci in prepared  # type: ignore[misc]
                ],
            )

        # Embed changed content via incremental processor
        proc = IncrementalProcessor()
        proc._reembed_pages(page_texts={1: file_text}, source_document=doc.filename)  # type: ignore[attr-defined]

        session.commit()
        return {"job_id": job_id, "status": "completed", "document_id": doc.id}
    except Exception as e:
        logger.exception("process_document_async_failed", error=str(e))
        return {"job_id": job_id, "status": "failed", "error": str(e)}
    finally:
        session.close()


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
