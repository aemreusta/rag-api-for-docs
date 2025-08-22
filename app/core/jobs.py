"""
Celery application setup for background processing.

This module defines the Celery app configured to use Redis as both the
broker and result backend. Tasks are defined in follow-up edits.
"""

from __future__ import annotations

import io
import os

try:
    from celery import Celery  # type: ignore

    _HAS_CELERY = True
except Exception:  # pragma: no cover
    Celery = None  # type: ignore
    _HAS_CELERY = False
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.dedup import ContentDeduplicator
from app.core.embedding_manager import QuotaError, get_embedding_manager
from app.core.incremental import IncrementalProcessor
from app.core.ingestion import NLTKAdaptiveChunker
from app.core.language_detection import detect_document_language
from app.core.logging_config import get_logger, set_request_id, set_trace_id, setup_logging
from app.core.metadata import ChunkMetadataExtractor
from app.core.request_tracking import get_request_tracker
from app.core.storage import get_storage_backend
from app.db.models import Document, ProcessingJob

# Ensure logging is configured for workers
setup_logging()
logger = get_logger(__name__)


def _create_session():
    """Create a SQLAlchemy session for worker context."""
    import os

    db_url = (
        os.getenv("DATABASE_URL")
        or getattr(settings, "DATABASE_URL", None)
        or (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_SERVER}:5432/{settings.POSTGRES_DB}"
        )
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=create_engine(db_url))
    return SessionLocal()


def _update_job_progress(
    session,
    job_id: str,
    progress_percent: int,
    status: str = None,
    result_data: dict = None,
    error_message: str = None,
) -> None:
    """Update job progress in the database."""
    try:
        from datetime import datetime, timezone

        job = session.get(ProcessingJob, job_id)
        if job:
            job.progress_percent = progress_percent

            if status:
                job.status = status

            if result_data:
                job.result_data = result_data

            if error_message:
                job.error_message = error_message

            if status in ("completed", "failed"):
                if status == "completed":
                    job.completed_at = datetime.now(timezone.utc)
                else:
                    job.error_message = error_message

            session.commit()

            logger.info(
                "Job progress updated",
                job_id=job_id,
                progress_percent=progress_percent,
                status=status,
            )
    except Exception as e:
        logger.warning(
            "Failed to update job progress",
            job_id=job_id,
            error=str(e),
        )


def _init_task_context_from_job(document_data: dict, job_id: str) -> tuple[str | None, str, str]:
    """Initialize request/trace context from job payload and return identifiers."""
    request_id = document_data.get("request_id")
    trace_id = document_data.get("trace_id", job_id)
    parent_operation = document_data.get("parent_operation", "unknown")

    if request_id:
        set_request_id(request_id)
    else:
        set_request_id()
    set_trace_id(trace_id)

    return request_id, trace_id, parent_operation


def _safe_retrieve_and_extract(doc: Document, storage_uri: str | None):
    """Retrieve file and extract content, returning safe defaults on failure."""
    try:
        return _retrieve_and_extract_content(doc, storage_uri)
    except Exception as err:  # noqa: BLE001 - want broad catch at task boundary
        logger.warning(
            "file_retrieval_failed",
            extra={
                "document_id": getattr(doc, "id", None),
                "storage_uri": storage_uri or getattr(doc, "storage_uri", None),
                "error": str(err),
            },
        )
        return "", {}, {"page_count": 0, "word_count": 0}


def _update_document_metadata_from_extraction(doc: Document, metadata: dict, during: str) -> None:
    """Update document page/word counts from extraction metadata with logs."""
    if not metadata:
        return
    if metadata.get("page_count") and doc.page_count != metadata["page_count"]:
        logger.info(
            f"Document page count updated during {during}",
            document_id=doc.id,
            old_page_count=doc.page_count,
            new_page_count=metadata["page_count"],
        )
        doc.page_count = metadata["page_count"]

    if metadata.get("word_count") and doc.word_count != metadata["word_count"]:
        logger.info(
            f"Document word count updated during {during}",
            document_id=doc.id,
            old_word_count=doc.word_count,
            new_word_count=metadata["word_count"],
        )
        doc.word_count = metadata["word_count"]


def _detect_and_update_language(doc: Document, file_text: str) -> None:
    """Detect language from text and update doc if changed."""
    if not file_text:
        return
    detected_language = detect_document_language(
        text=file_text, filename=doc.filename or "", default="tr"
    )
    if doc.language != detected_language:
        logger.info(
            "Document language updated",
            document_id=doc.id,
            filename=doc.filename,
            old_language=doc.language,
            detected_language=detected_language,
        )
        doc.language = detected_language


def _mark_status(session, doc: Document, status_value: str) -> None:
    """Mark document status safely without raising errors."""
    try:
        from app.db.models import DocumentStatusEnum

        db_doc = session.get(Document, doc.id)
        if db_doc:
            db_doc.status = getattr(DocumentStatusEnum, status_value)
    except Exception:
        pass


def _extract_pdf_per_page(file_bytes: bytes) -> tuple[str, dict[int, str], dict]:
    """Extract text from PDF per-page using pypdf with metadata.

    Returns:
        tuple: (combined_text, page_texts_dict, metadata_dict)
            - combined_text: All text combined
            - page_texts_dict: {page_num: text}
            - metadata_dict: {page_count: int, word_count: int}
    """
    try:
        import pypdf

        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        page_texts = {}
        combined_texts = []
        total_word_count = 0

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_texts[page_num] = page_text
                    combined_texts.append(page_text)

                    # Count words for this page
                    page_word_count = len(page_text.split())
                    total_word_count += page_word_count

                    logger.debug(
                        "pdf_page_extracted",
                        extra={
                            "page_number": page_num,
                            "text_length": len(page_text),
                            "char_count": len(page_text.strip()),
                            "word_count": page_word_count,
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
        page_count = len(pdf_reader.pages)

        metadata = {
            "page_count": page_count,
            "word_count": total_word_count,
        }

        logger.info(
            "pdf_extraction_completed",
            extra={
                "total_pages": page_count,
                "extracted_pages": len(page_texts),
                "combined_length": len(combined_text),
                "total_word_count": total_word_count,
            },
        )

        return combined_text, page_texts, metadata

    except Exception as e:
        logger.error(
            "pdf_extraction_failed", extra={"error": str(e), "error_type": type(e).__name__}
        )
        # Fallback to decode attempt
        try:
            fallback_text = file_bytes.decode("utf-8", errors="ignore")
            word_count = len(fallback_text.split()) if fallback_text else 0
            metadata = {"page_count": 1, "word_count": word_count}
            return fallback_text, {1: fallback_text} if fallback_text else {}, metadata
        except Exception:
            return "", {}, {"page_count": 0, "word_count": 0}


def _create_celery_app():
    """Create and configure the Celery app instance."""
    if not _HAS_CELERY:
        # Lightweight stub when Celery is not installed (CI/local unit tests)
        class _Stub:
            def task(self, *args, **kwargs):  # noqa: D401 - simple decorator stub
                def decorator(fn):
                    return fn

                return decorator

        logger.warning("Celery not installed; using no-op task decorators for tests")
        return _Stub()
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


# Celery application singleton used by worker and beat (or stub in tests)
celery_app = _create_celery_app()


def _retrieve_and_extract_content(
    doc: Document, storage_uri: str | None
) -> tuple[str, dict[int, str], dict]:
    """Retrieve file and extract content based on MIME type with metadata.

    Returns:
        tuple: (file_text, page_texts, metadata)
    """
    backend = get_storage_backend()
    file_bytes = backend.retrieve_file(storage_uri or doc.storage_uri)

    # Handle PDF extraction per-page using pypdf
    if doc.mime_type == "application/pdf":
        return _extract_pdf_per_page(file_bytes)
    else:
        # Fallback for non-PDF files
        file_text = file_bytes.decode("utf-8", errors="ignore")
        page_texts = {1: file_text} if file_text else {}
        word_count = len(file_text.split()) if file_text else 0
        metadata = {"page_count": 1, "word_count": word_count}
        return file_text, page_texts, metadata


def _process_chunks_and_embeddings(
    session, doc: Document, file_text: str, page_texts: dict[int, str], task_context=None
) -> None:
    """Process chunks and generate embeddings."""
    import time

    start_time = time.time()

    logger.info(
        "Starting chunk processing",
        document_id=doc.id,
        filename=doc.filename,
        text_length=len(file_text) if file_text else 0,
        page_count=len(page_texts),
    )

    # Chunk using the same chunker implementation as ingestion
    chunker = NLTKAdaptiveChunker()
    chunks = chunker.chunk(file_text) if file_text else []

    logger.info(
        "Text chunking completed",
        document_id=doc.id,
        chunk_count=len(chunks),
        avg_chunk_length=(
            round(sum(len(chunk) for chunk in chunks) / len(chunks), 1) if chunks else 0
        ),
    )

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
        chunk_start = time.time()
        dedup.upsert_chunks(
            session,
            document_id=doc.id,
            chunks=[
                # Build ChunkInput without importing in worker context
                type("CI", (), ci)()
                for ci in prepared  # type: ignore[misc]
            ],
        )
        chunk_duration = time.time() - chunk_start

        logger.info(
            "Chunks upserted",
            document_id=doc.id,
            prepared_chunk_count=len(prepared),
            upsert_time_seconds=round(chunk_duration, 3),
        )

    # Embed changed content via incremental processor with quota handling
    # Log embedding provider/model for observability
    try:
        embedding_manager = get_embedding_manager()
        provider_status = embedding_manager.get_provider_status()

        logger.info(
            "embedding_providers_status", provider_status=provider_status, document_id=doc.id
        )

        # Check if any provider is available
        available_providers = [p for p, s in provider_status.items() if s.get("available", False)]
        if not available_providers:
            # Calculate minimum wait time
            min_wait = min(
                [
                    s.get("wait_seconds", 0)
                    for s in provider_status.values()
                    if s.get("wait_seconds", 0) > 0
                ]
                or [3600]
            )
            raise QuotaError(f"All embedding providers quota exhausted. Minimum wait: {min_wait}s")

    except Exception as e:
        logger.warning(
            "Failed to get embedding provider status", error=str(e), error_type=type(e).__name__
        )

    embed_start = time.time()

    try:
        proc = IncrementalProcessor()
        # Pass the actual per-page texts extracted from PDF or fallback to single page
        proc._reembed_pages(page_texts=page_texts or {1: file_text}, source_document=doc.filename)  # type: ignore[attr-defined]
        embed_duration = time.time() - embed_start

        logger.info(
            "Embedding generation completed successfully",
            document_id=doc.id,
            embedding_time_seconds=round(embed_duration, 3),
        )

    except QuotaError as e:
        logger.warning(
            "Embedding generation failed due to quota exhaustion", document_id=doc.id, error=str(e)
        )
        # Raise quota error - will be handled by the calling task
        if task_context:
            raise task_context.retry(countdown=1800, exc=e) from e  # Retry in 30 minutes
        else:
            raise e
    except Exception as e:
        embed_duration = time.time() - embed_start
        logger.error(
            "Embedding generation failed",
            document_id=doc.id,
            error=str(e),
            error_type=type(e).__name__,
            embedding_time_seconds=round(embed_duration, 3),
        )
        raise

    total_duration = time.time() - start_time

    logger.info(
        "Document processing completed",
        document_id=doc.id,
        filename=doc.filename,
        total_processing_time_seconds=round(total_duration, 3),
        embedding_time_seconds=round(embed_duration, 3),
        chunk_count=len(chunks),
        page_count=len(page_texts),
    )


@celery_app.task(bind=True, max_retries=3, name="jobs.process_document_async")
def process_document_async(self, job_id: str, document_data: dict) -> dict:
    """Background document processing task with comprehensive tracking.

    Minimal v1: chunk via ingestion engine, dedup/update chunks, embed via IncrementalProcessor.
    """
    # Initialize correlation IDs for worker task context
    request_id, trace_id, parent_operation = _init_task_context_from_job(document_data, job_id)

    # Initialize request tracker
    tracker = get_request_tracker()

    logger.info(
        "Document processing task started",
        job_id=job_id,
        request_id=request_id,
        trace_id=trace_id,
        parent_operation=parent_operation,
        payload_keys=list(document_data.keys()),
        worker_pid=os.getpid(),
    )
    # DB session (worker-safe)
    session = _create_session()
    try:
        # Mark job as started
        _update_job_progress(session, job_id, 5, "processing", {"stage": "initializing"})

        doc_id = document_data.get("document_id")
        storage_uri = document_data.get("storage_uri")
        # In minimal/eager test mode, allow empty payload and just report queued
        if not doc_id:
            return {"job_id": job_id, "status": "queued"}
        doc = session.get(Document, doc_id)
        if not doc:
            _update_job_progress(session, job_id, 0, "failed", None, "document_not_found")
            return {"job_id": job_id, "status": "failed", "error": "document_not_found"}

        # Wrap main processing with comprehensive tracking
        with tracker.track_document_processing(
            document_id=doc.id, stage="full_processing", filename=doc.filename
        ) as context:
            # Fetch file contents and extract text with metadata
            _update_job_progress(session, job_id, 25, "processing", {"stage": "extracting_content"})
            file_text, page_texts, metadata = _safe_retrieve_and_extract(doc, storage_uri)

            # Update document metadata
            _update_document_metadata_from_extraction(doc, metadata, during="processing")

            # Detect and update document language
            _detect_and_update_language(doc, file_text)
            _update_job_progress(session, job_id, 40, "processing", {"stage": "processing_chunks"})

            # Process chunks and embeddings
            _update_job_progress(
                session, job_id, 60, "processing", {"stage": "generating_embeddings"}
            )
            _process_chunks_and_embeddings(session, doc, file_text, page_texts, task_context=self)
            _update_job_progress(session, job_id, 90, "processing", {"stage": "finalizing"})

            # Add processing metrics to context
            context.update(
                {
                    "page_count": metadata.get("page_count", 0),
                    "word_count": metadata.get("word_count", 0),
                    "text_length": len(file_text) if file_text else 0,
                    "page_texts_count": len(page_texts),
                    "detected_language": doc.language,
                }
            )

        # Mark document completed
        _mark_status(session, doc, "COMPLETED")
        session.commit()

        # Mark job as completed
        _update_job_progress(
            session,
            job_id,
            100,
            "completed",
            {
                "document_id": doc.id,
                "request_id": request_id,
                "trace_id": trace_id,
            },
        )

        logger.info(
            "Document processing task completed successfully",
            job_id=job_id,
            document_id=doc.id,
            filename=doc.filename,
            request_id=request_id,
            trace_id=trace_id,
            parent_operation=parent_operation,
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "document_id": doc.id,
            "request_id": request_id,
            "trace_id": trace_id,
        }
    except Exception as e:
        # Mark job as failed
        _update_job_progress(session, job_id, 0, "failed", None, str(e))

        logger.error(
            "Document processing task failed",
            job_id=job_id,
            document_id=doc_id,
            request_id=request_id,
            trace_id=trace_id,
            parent_operation=parent_operation,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "request_id": request_id,
            "trace_id": trace_id,
        }
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=5, retry_backoff=True, name="jobs.scrape_url_async")
def scrape_url_async(self, job_id: str, url: str, options: dict | None = None) -> dict:
    """Background URL scraping task (stub)."""
    # Handle correlation IDs if provided in options
    request_id = options.get("request_id") if options else None
    trace_id = options.get("trace_id", job_id) if options else job_id
    parent_operation = (
        options.get("parent_operation", "url_scraping") if options else "url_scraping"
    )

    # Initialize correlation IDs for worker task context
    if request_id:
        set_request_id(request_id)
    else:
        set_request_id()
    set_trace_id(trace_id)

    logger.info(
        "URL scraping task started",
        job_id=job_id,
        url=url,
        request_id=request_id,
        trace_id=trace_id,
        parent_operation=parent_operation,
        options=options or {},
        worker_pid=os.getpid(),
    )

    return {"job_id": job_id, "status": "queued", "request_id": request_id, "trace_id": trace_id}


@celery_app.task(bind=True, max_retries=3, name="jobs.retry_failed_document")
def retry_failed_document(self, document_id: str, retry_reason: str = "manual_retry") -> dict:
    """Retry processing for a failed document.

    Args:
        document_id: ID of the document to retry
        retry_reason: Reason for the retry (for logging)

    Returns:
        dict: Processing result
    """
    # Initialize correlation IDs for retry task context
    request_id = set_request_id()
    set_trace_id(request_id)

    logger.info(
        "Document retry task started",
        document_id=document_id,
        retry_reason=retry_reason,
        worker_pid=os.getpid(),
    )

    # DB session (worker-safe)
    session = _create_session()

    try:
        doc = session.get(Document, document_id)
        if not doc:
            return {"status": "failed", "error": "document_not_found"}

        # Reset document status to processing
        from app.db.models import DocumentStatusEnum

        doc.status = DocumentStatusEnum.PROCESSING
        session.flush()

        logger.info(
            "Document status reset to processing",
            document_id=document_id,
            retry_reason=retry_reason,
        )

        # Fetch file contents and extract text with metadata
        file_text, page_texts, metadata = _safe_retrieve_and_extract(doc, None)

        # Update document metadata
        _update_document_metadata_from_extraction(doc, metadata, during="retry")

        # Detect and update document language
        _detect_and_update_language(doc, file_text)

        # Process chunks and embeddings
        _process_chunks_and_embeddings(session, doc, file_text, page_texts, task_context=None)

        # Mark document completed
        _mark_status(session, doc, "COMPLETED")
        session.commit()

        logger.info(
            "Document retry task completed successfully",
            document_id=document_id,
            retry_reason=retry_reason,
            filename=doc.filename,
        )

        return {
            "status": "completed",
            "document_id": document_id,
            "retry_reason": retry_reason,
        }

    except Exception as e:
        # Mark document as failed
        try:
            db_doc = session.get(Document, document_id)
            if db_doc:
                _mark_status(session, db_doc, "FAILED")
            session.commit()
        except Exception:
            pass

        logger.error(
            "Document retry task failed",
            document_id=document_id,
            retry_reason=retry_reason,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(e),
        }
    finally:
        session.close()


@celery_app.task(name="jobs.cleanup_failed_jobs")
def cleanup_failed_jobs() -> int:
    """Periodic cleanup task (stub). Returns number of cleaned items."""
    logger.info("cleanup_failed_jobs invoked")
    # Placeholder count until real job store is implemented
    return 0
