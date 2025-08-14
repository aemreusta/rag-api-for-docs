"""
Celery application setup for background processing.

This module defines the Celery app configured to use Redis as both the
broker and result backend. Tasks are defined in follow-up edits.
"""

from __future__ import annotations

import io

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.dedup import ContentDeduplicator
from app.core.embeddings import get_embedding_model
from app.core.incremental import IncrementalProcessor
from app.core.ingestion import NLTKAdaptiveChunker
from app.core.logging_config import get_logger, set_request_id, set_trace_id, setup_logging
from app.core.metadata import ChunkMetadataExtractor
from app.core.storage import get_storage_backend
from app.db.models import Document

# Ensure logging is configured for workers
setup_logging()
logger = get_logger(__name__)


def _extract_pdf_per_page(file_bytes: bytes) -> tuple[str, dict[int, str]]:
    """Extract text from PDF per-page using pypdf.

    Returns:
        tuple: (combined_text, page_texts_dict) where page_texts_dict is {page_num: text}
    """
    try:
        import pypdf

        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        page_texts = {}
        combined_texts = []

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_texts[page_num] = page_text
                    combined_texts.append(page_text)
                    logger.debug(
                        "pdf_page_extracted",
                        extra={
                            "page_number": page_num,
                            "text_length": len(page_text),
                            "char_count": len(page_text.strip()),
                        },
                    )
                else:
                    logger.warning("pdf_page_empty", extra={"page_number": page_num})
            except Exception as e:
                logger.warning(
                    "pdf_page_extraction_failed", extra={"page_number": page_num, "error": str(e)}
                )
                continue

        combined_text = "\n\n".join(combined_texts)

        logger.info(
            "pdf_extraction_completed",
            extra={
                "total_pages": len(pdf_reader.pages),
                "extracted_pages": len(page_texts),
                "combined_length": len(combined_text),
            },
        )

        return combined_text, page_texts

    except Exception as e:
        logger.error(
            "pdf_extraction_failed", extra={"error": str(e), "error_type": type(e).__name__}
        )
        # Fallback to decode attempt
        try:
            fallback_text = file_bytes.decode("utf-8", errors="ignore")
            return fallback_text, {1: fallback_text} if fallback_text else {}
        except Exception:
            return "", {}


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


def _retrieve_and_extract_content(
    doc: Document, storage_uri: str | None
) -> tuple[str, dict[int, str]]:
    """Retrieve file and extract content based on MIME type."""
    backend = get_storage_backend()
    file_bytes = backend.retrieve_file(storage_uri or doc.storage_uri)

    # Handle PDF extraction per-page using pypdf
    if doc.mime_type == "application/pdf":
        return _extract_pdf_per_page(file_bytes)
    else:
        # Fallback for non-PDF files
        file_text = file_bytes.decode("utf-8", errors="ignore")
        page_texts = {1: file_text} if file_text else {}
        return file_text, page_texts


def _process_chunks_and_embeddings(
    session, doc: Document, file_text: str, page_texts: dict[int, str]
) -> None:
    """Process chunks and generate embeddings."""
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
    # Log embedding provider/model for observability
    try:
        embedder = get_embedding_model()
        logger.info(
            "embedding_provider_selected",
            extra={
                "provider": type(embedder).__name__,
                "model": getattr(embedder, "model_name", getattr(embedder, "model", None)),
                "dim": getattr(embedder, "embed_dim", None)
                or getattr(embedder, "dimensions", None),
            },
        )
    except Exception:
        pass
    proc = IncrementalProcessor()
    # Pass the actual per-page texts extracted from PDF or fallback to single page
    proc._reembed_pages(page_texts=page_texts or {1: file_text}, source_document=doc.filename)  # type: ignore[attr-defined]


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

        # Fetch file contents and extract text
        try:
            file_text, page_texts = _retrieve_and_extract_content(doc, storage_uri)
        except Exception as e:
            logger.warning(
                "file_retrieval_failed",
                extra={
                    "document_id": doc_id,
                    "storage_uri": storage_uri or doc.storage_uri,
                    "error": str(e),
                },
            )
            file_text = ""
            page_texts = {}

        # Process chunks and embeddings
        _process_chunks_and_embeddings(session, doc, file_text, page_texts)

        # Mark document completed
        try:
            db_doc = session.get(Document, doc.id)
            if db_doc:
                from app.db.models import DocumentStatusEnum

                db_doc.status = DocumentStatusEnum.COMPLETED  # type: ignore[assignment]
        except Exception:
            pass
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
