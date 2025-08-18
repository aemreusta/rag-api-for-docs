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
from app.core.jobs import celery_app, process_document_async
from app.core.logging_config import get_logger
from app.core.metrics import get_metrics_backend
from app.core.quality import QualityAssurance
from app.core.storage import get_storage_backend
from app.db.models import Document, DocumentStatusEnum

router = APIRouter(
    prefix="/docs", tags=["Documents"], responses={404: {"description": "Not found"}}
)
logger = get_logger(__name__)
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
    pass


# Use a module-level singleton default to satisfy ruff B008
UPLOAD_FILE_FORM = File(...)


DB_DEP_UPLOAD: Session = Depends(get_db_session)


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

    # Persist file to local storage and upsert document via deduplicator
    def _safe_filename(name: str) -> str:
        base = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-") or "upload"
        return base[:200]

    payload = await file.read()
    content_hash = compute_sha256_hex(payload)
    safe_name = _safe_filename(file.filename)
    # Cache check: if a document with this content_hash was recently processed, reuse storage_uri
    try:
        cache = await get_cache_backend()
        cache_key = f"doc_hash:{content_hash}"
        cached_uri = await cache.get(cache_key)
        if isinstance(cached_uri, str) and cached_uri:
            storage_uri = cached_uri
        else:
            storage_uri = storage.store_file(payload, safe_name)
            await cache.set(cache_key, storage_uri, ttl=600)
    except Exception:
        storage_uri = storage.store_file(payload, safe_name)
    dedup = ContentDeduplicator()
    doc, _ = dedup.upsert_document_by_hash(
        db,
        filename=file.filename,
        storage_uri=storage_uri,
        mime_type=file.content_type or "application/octet-stream",
        file_size=len(payload),
        page_count=None,
        new_content_hash=content_hash,
    )

    # Update DB-backed status to processing immediately after enqueue
    try:
        db_doc = db.get(Document, doc.id)
        if db_doc:
            db_doc.status = DocumentStatusEnum.PROCESSING  # type: ignore[assignment]
            db.commit()
    except Exception:
        pass

    # Enqueue background processing
    try:
        job_id = str(uuid.uuid4())
        process_document_async.delay(job_id, {"document_id": doc.id, "storage_uri": storage_uri})
        logger.info(
            "upload_enqueued", extra={"doc_id": doc.id, "filename": file.filename, "job_id": job_id}
        )
    except Exception:
        logger.warning("failed_to_enqueue_background_job", extra={"doc_id": doc.id})
    metrics.increment_counter("vector_search_requests_total", {"status": "ingest"})
    return DocumentDetail(id=doc.id, filename=file.filename, status="processing")


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
    detail: dict | None = None


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Get background job status by id.",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Map Celery job states to API status semantics."""
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
        detail = None
        try:
            info = res.info  # type: ignore[attr-defined]
            if isinstance(info, dict):
                detail = info
        except Exception:
            detail = None
        return JobStatusResponse(id=job_id, status=status, detail=detail)
    except Exception:
        return JobStatusResponse(id=job_id, status="pending", detail=None)
