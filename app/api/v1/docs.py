from __future__ import annotations

import re
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.cache import get_cache_backend
from app.core.dedup import ContentDeduplicator, compute_sha256_hex
from app.core.incremental import ChangeSet, IncrementalProcessor
from app.core.jobs import celery_app, process_document_async, retry_failed_document
from app.core.logging_config import get_logger, get_request_id, get_trace_id
from app.core.metrics import get_metrics_backend
from app.core.quality import QualityAssurance
from app.core.storage import get_storage_backend
from app.db.models import Document, DocumentStatusEnum, ProcessingJob

logger = get_logger(__name__)

router = APIRouter(
    prefix="/docs", tags=["Documents"], responses={404: {"description": "Not found"}}
)
metrics = get_metrics_backend()

storage = get_storage_backend()


# Remove in-memory storage; use DB-backed status


class DocumentSummary(BaseModel):
    id: str
    filename: str
    status: Literal["pending", "processing", "completed", "failed"]


class DocumentStatus(BaseModel):
    id: str
    status: Literal["pending", "processing", "completed", "failed"]


class DocumentDetail(DocumentSummary):
    job_id: str | None = None


# Use a module-level singleton default to satisfy ruff B008
UPLOAD_FILE_FORM = File(...)


DB_DEP_UPLOAD: Session = Depends(get_db_session)


def _safe_filename(name: str) -> str:
    """Generate a safe filename by sanitizing and truncating."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-") or "upload"
    return base[:200]


async def _detect_language_from_filename(filename: str) -> str:
    """Detect language from filename, fallback to Turkish."""
    from app.core.language_detection import LanguageDetector

    detector = LanguageDetector(default_language="tr")
    detected_lang = detector.detect_language_from_filename(filename)
    if detected_lang:
        logger.info(
            "Language detected from filename during upload",
            filename=filename,
            detected_language=detected_lang,
        )
        return detected_lang
    return "tr"


async def _get_or_store_file(payload: bytes, safe_name: str, content_hash: str) -> str:
    """Get storage URI from cache or store file and cache the URI.

    Uses content-addressed storage to prevent duplicate file storage.
    """
    try:
        cache = await get_cache_backend()
        cache_key = f"doc_hash:{content_hash}"
        cached_uri = await cache.get(cache_key)
        if isinstance(cached_uri, str) and cached_uri:
            return cached_uri

        # Use content-addressed storage with content_hash
        storage_uri = storage.store_file(payload, safe_name, content_hash)
        await cache.set(cache_key, storage_uri, ttl=600)
        return storage_uri
    except Exception:
        # Fallback without content_hash for backward compatibility
        return storage.store_file(payload, safe_name)


def _update_document_status(db: Session, doc_id: str) -> None:
    """Update document status to processing."""
    try:
        db_doc = db.get(Document, doc_id)
        if db_doc:
            db_doc.status = DocumentStatusEnum.PROCESSING  # type: ignore[assignment]
            db.commit()
    except Exception:
        pass


def _enqueue_processing_job(doc_id: str, storage_uri: str, filename: str, db: Session) -> str:
    """Enqueue background processing job with progress tracking."""
    try:
        job_id = str(uuid.uuid4())
        request_id = get_request_id()
        trace_id = get_trace_id()

        job_data = {
            "document_id": doc_id,
            "storage_uri": storage_uri,
            "request_id": request_id,
            "trace_id": trace_id,
            "parent_operation": "document_upload",
        }

        # Create ProcessingJob record in database for progress tracking
        from datetime import datetime, timezone

        processing_job = ProcessingJob(
            id=job_id,
            job_type="upload",
            status="pending",
            input_data=job_data,
            progress_percent=0,
            created_at=datetime.now(timezone.utc),
            created_by=request_id or "system",
        )
        db.add(processing_job)
        db.commit()

        # Enqueue the background task
        process_document_async.delay(job_id, job_data)

        logger.info(
            "Document upload successful, processing job enqueued",
            doc_id=doc_id,
            filename=filename,
            job_id=job_id,
            request_id=request_id,
            trace_id=trace_id,
        )

        return job_id

    except Exception as e:
        logger.error(
            "Failed to enqueue background job",
            doc_id=doc_id,
            filename=filename,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


@router.post(
    "/upload",
    response_model=DocumentDetail,
    status_code=201,
    summary="Upload a document",
    response_description="Document enqueued for processing",
    description=(
        "Upload a document for ingestion.\n\n"
        "- Accepts: multipart/form-data with `file`\n"
        "- Validates file size and type\n"
        "- Stores file using configured storage backend (local or MinIO)\n"
        "- Deduplicates by content hash; bumps version on change\n"
        "- Enqueues background processing job\n\n"
        "Returns basic document metadata and initial status.\n\n"
        "Example cURL:\n\n"
        'curl -X POST -F "file=@mydoc.pdf" http://localhost:8000/api/v1/docs/upload'
    ),
)
async def upload_document(
    file: UploadFile = UPLOAD_FILE_FORM, db: Session = DB_DEP_UPLOAD
) -> DocumentDetail:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    # Basic validation
    validation = await QualityAssurance.validate_upload(file)
    if not validation.ok:
        logger.info(
            "upload_validation_failed",
            extra={"filename": file.filename, "reason": validation.reason},
        )
        raise HTTPException(status_code=400, detail=f"Invalid upload: {validation.reason}")

    # Process file data
    payload = await file.read()
    content_hash = compute_sha256_hex(payload)
    safe_name = _safe_filename(file.filename)

    # Detect language and get storage URI
    initial_language = (
        await _detect_language_from_filename(file.filename) if file.filename else "tr"
    )
    storage_uri = await _get_or_store_file(payload, safe_name, content_hash)

    # Create document record
    dedup = ContentDeduplicator()
    doc, _ = dedup.upsert_document_by_hash(
        db,
        filename=file.filename,
        storage_uri=storage_uri,
        mime_type=file.content_type or "application/octet-stream",
        file_size=len(payload),
        page_count=None,
        new_content_hash=content_hash,
        language=initial_language,
    )

    # Update status and enqueue processing
    _update_document_status(db, doc.id)
    job_id = _enqueue_processing_job(doc.id, storage_uri, file.filename, db)
    metrics.increment_counter("vector_search_requests_total", {"status": "ingest"})
    return DocumentDetail(id=doc.id, filename=file.filename, status="processing", job_id=job_id)


DB_DEP_LIST: Session = Depends(get_db_session)


def _status_literal(s: object) -> str:
    try:
        if isinstance(s, DocumentStatusEnum):
            return s.value
        txt = str(s)
        if txt.startswith("DocumentStatusEnum."):
            return txt.split(".", 1)[1].lower()
        return txt.lower()
    except Exception:
        return "pending"


@router.get(
    "",
    response_model=list[DocumentSummary],
    summary="List documents",
    response_description="Summary list of documents",
)
async def list_documents(db: Session = DB_DEP_LIST) -> list[DocumentSummary]:
    rows = db.query(Document).all()
    return [
        DocumentSummary(id=r.id, filename=r.filename, status=_status_literal(r.status))
        for r in rows
    ]


DB_DEP_GET: Session = Depends(get_db_session)


@router.get(
    "/{doc_id}",
    response_model=DocumentDetail,
    summary="Get document",
    response_description="Document details",
    description="Get document metadata by id.",
)
async def get_document(doc_id: str, db: Session = DB_DEP_GET) -> DocumentDetail:
    r = db.get(Document, doc_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(id=r.id, filename=r.filename, status=_status_literal(r.status))


DB_DEP_STATUS: Session = Depends(get_db_session)


@router.get(
    "/status/{doc_id}",
    response_model=DocumentStatus,
    summary="Get document status",
    response_description="Document status",
    description="Get ingestion status of a document by id.",
)
async def get_document_status(doc_id: str, db: Session = DB_DEP_STATUS) -> DocumentStatus:
    r = db.get(Document, doc_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatus(id=r.id, status=_status_literal(r.status))


# Optional endpoint: URL scraping prototype
class ScrapeRequest(BaseModel):
    url: str


class ScrapeResponse(BaseModel):
    id: str
    url: str
    status: Literal["pending", "processing", "completed", "failed"]


@router.post("/scrape", response_model=ScrapeResponse, status_code=202)
async def scrape_url(payload: ScrapeRequest) -> ScrapeResponse:
    job_id = str(uuid.uuid4())
    # Real impl would enqueue scraping task; here we simulate accepted state
    return ScrapeResponse(id=job_id, url=payload.url, status="pending")


# ----------------------------
# Incremental update endpoints
# ----------------------------


class PageHashes(BaseModel):
    pages: dict[int, str]


class PageTexts(BaseModel):
    pages: dict[int, str]


class DetectChangesRequest(BaseModel):
    document_id: str
    new_content_hash: str
    page_hashes: PageHashes | None = None


class DetectChangesResponse(BaseModel):
    is_new_version: bool
    changed_pages: list[int]
    reason: str | None = None


DB_DEP_CHANGES: Session = Depends(get_db_session)


@router.post(
    "/detect-changes",
    response_model=DetectChangesResponse,
    summary="Detect changes",
    description=(
        "Detect changed pages compared to current stored version.\n\n"
        "Provide a `document_id`, a new `content_hash`, and optionally per-page hashes."
    ),
)
async def detect_changes(
    payload: DetectChangesRequest, db: Session = DB_DEP_CHANGES
) -> DetectChangesResponse:
    proc = IncrementalProcessor()
    changes = proc.detect_changes(
        db=db,
        document_id=payload.document_id,
        new_content_hash=payload.new_content_hash,
        new_page_hashes=(payload.page_hashes.pages if payload.page_hashes else None),
    )
    return DetectChangesResponse(
        is_new_version=changes.is_new_version,
        changed_pages=changes.changed_pages,
        reason=changes.reason,
    )


class ApplyChangesRequest(BaseModel):
    document_id: str
    changes: DetectChangesResponse
    new_content_hash: str | None = None
    page_texts: PageTexts


class ApplyChangesResponse(BaseModel):
    updated_pages: list[int]
    version: int | None = None


DB_DEP: Session = Depends(get_db_session)


@router.put(
    "/apply-changes",
    response_model=ApplyChangesResponse,
    status_code=200,
    summary="Apply changes",
    description=(
        "Apply detected changes: re-embed changed pages and upsert chunks.\n\n"
        "- Re-embeds only changed pages\n"
        "- Upserts chunk rows with page metadata\n"
        "- Bumps document version and updates content hash"
    ),
)
async def apply_changes(payload: ApplyChangesRequest, db: Session = DB_DEP) -> ApplyChangesResponse:
    proc = IncrementalProcessor()
    # Convert response-model shape back into ChangeSet
    change_set = ChangeSet(
        is_new_version=payload.changes.is_new_version,
        changed_pages=payload.changes.changed_pages,
        reason=payload.changes.reason,
    )
    proc.process_incremental_update(
        db=db,
        document_id=payload.document_id,
        changes=change_set,
        new_content_hash=payload.new_content_hash,
        new_page_texts=payload.page_texts.pages,
    )
    # Fetch version to report
    from app.db.models import Document

    version = None
    try:
        doc = db.get(Document, payload.document_id)
        version = doc.version if doc else None
    except Exception:
        version = None
    return ApplyChangesResponse(updated_pages=payload.changes.changed_pages, version=version)


# ----------------------------
# Jobs status endpoint
# ----------------------------


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["pending", "processing", "completed", "failed", "retry"]
    progress_percent: int | None = None
    detail: dict | None = None


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Get background job status by id with progress tracking.",
)
async def get_job_status(job_id: str, db: Session = Depends(get_db_session)) -> JobStatusResponse:  # noqa: C901,B008
    """Get job status with progress tracking from both Celery and database."""
    progress_percent = None
    detail = None

    # Query processing_jobs table for progress information
    try:
        processing_job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if processing_job:
            progress_percent = processing_job.progress_percent
            if processing_job.result_data:
                detail = processing_job.result_data
            elif processing_job.error_message:
                detail = {"error": processing_job.error_message}
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning("Failed to query processing_jobs table", job_id=job_id, error=str(e))

    # Get Celery job state
    try:
        res = celery_app.AsyncResult(job_id)
        state = (res.state or "PENDING").upper()

        if state in {"PENDING"}:
            status = "pending"
        elif state in {"RECEIVED", "STARTED"}:
            status = "processing"
        elif state in {"RETRY"}:
            status = "retry"
        elif state in {"FAILURE"}:
            status = "failed"
        else:  # SUCCESS or others
            status = "completed"

        # Use Celery result info if no detail from database
        if detail is None:
            try:
                info = res.info  # type: ignore[attr-defined]
                if isinstance(info, dict):
                    detail = info
            except Exception:
                detail = None

    except Exception:
        status = "pending"
        detail = None

    return JobStatusResponse(
        id=job_id, status=status, progress_percent=progress_percent, detail=detail
    )


# ----------------------------
# Document retry endpoints
# ----------------------------


class RetryDocumentResponse(BaseModel):
    job_id: str
    document_id: str
    status: Literal["queued", "processing", "failed"]
    message: str


@router.post(
    "/retry/{document_id}",
    response_model=RetryDocumentResponse,
    status_code=202,
    summary="Retry failed document processing",
    description=(
        "Retry processing for a failed document.\n\n"
        "- Resets document status to processing\n"
        "- Extracts metadata (page count, word count)\n"
        "- Detects proper language with Turkish default\n"
        "- Regenerates chunks and embeddings\n"
        "- Returns job ID for tracking retry status"
    ),
)
async def retry_document_processing(
    document_id: str, db: Session = DB_DEP
) -> RetryDocumentResponse:
    """Retry processing for a failed document."""
    # Check if document exists and can be retried
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if document is in a retryable state
    if doc.status not in [DocumentStatusEnum.FAILED, DocumentStatusEnum.PAUSED]:
        message = (
            f"Document status '{doc.status.value}' is not retryable. "
            "Only failed or paused documents can be retried."
        )
        raise HTTPException(status_code=400, detail=message)

    try:
        # Generate new job ID for the retry
        job_id = str(uuid.uuid4())

        # Enqueue retry task
        retry_failed_document.delay(document_id, "manual_api_retry")

        logger.info(
            "Document retry enqueued",
            document_id=document_id,
            job_id=job_id,
            previous_status=doc.status.value,
            request_id=get_request_id(),
            trace_id=get_trace_id(),
        )

        return RetryDocumentResponse(
            job_id=job_id,
            document_id=document_id,
            status="queued",
            message="Document retry has been queued for processing",
        )

    except Exception as e:
        logger.error(
            "Failed to enqueue document retry",
            document_id=document_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500, detail="Failed to enqueue document for retry processing"
        ) from e


class RetryFailedDocumentsResponse(BaseModel):
    queued_count: int
    job_ids: list[str]
    message: str


@router.post(
    "/retry-failed",
    response_model=RetryFailedDocumentsResponse,
    status_code=202,
    summary="Retry all failed documents",
    description=(
        "Batch retry processing for all failed documents.\n\n"
        "- Finds all documents with 'failed' status\n"
        "- Enqueues retry jobs for each failed document\n"
        "- Returns count and job IDs for tracking"
    ),
)
async def retry_all_failed_documents(
    db: Session = DB_DEP,
) -> RetryFailedDocumentsResponse:
    """Retry processing for all failed documents."""
    # Find all failed documents
    failed_docs = db.query(Document).filter(Document.status == DocumentStatusEnum.FAILED).all()

    if not failed_docs:
        return RetryFailedDocumentsResponse(
            queued_count=0, job_ids=[], message="No failed documents found to retry"
        )

    job_ids = []
    queued_count = 0

    for doc in failed_docs:
        try:
            job_id = str(uuid.uuid4())
            retry_failed_document.delay(doc.id, "batch_api_retry")
            job_ids.append(job_id)
            queued_count += 1

            logger.info(
                "Failed document queued for retry",
                document_id=doc.id,
                filename=doc.filename,
                job_id=job_id,
            )

        except Exception as e:
            logger.error(
                "Failed to enqueue document retry",
                document_id=doc.id,
                filename=doc.filename,
                error=str(e),
                error_type=type(e).__name__,
            )
            continue

    logger.info(
        "Batch retry completed",
        total_failed=len(failed_docs),
        queued_count=queued_count,
        failed_to_queue=len(failed_docs) - queued_count,
    )

    return RetryFailedDocumentsResponse(
        queued_count=queued_count,
        job_ids=job_ids,
        message=f"Queued {queued_count} failed documents for retry processing",
    )
