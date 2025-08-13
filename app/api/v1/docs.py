from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.dedup import ContentDeduplicator, compute_sha256_hex
from app.core.incremental import ChangeSet, IncrementalProcessor
from app.core.jobs import process_document_async
from app.core.logging_config import get_logger
from app.core.metrics import get_metrics_backend
from app.core.quality import QualityAssurance
from app.core.storage import get_storage_backend

router = APIRouter(prefix="/docs", tags=["Documents"])
logger = get_logger(__name__)
metrics = get_metrics_backend()

storage = get_storage_backend()


# In-memory storage for prototype behavior (to be replaced by DB integration)
@dataclass
class DocumentRecord:
    id: str
    filename: str
    status: Literal["pending", "processing", "completed", "failed"] = "pending"


_DOCUMENTS: dict[str, DocumentRecord] = {}


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


@router.post("/upload", response_model=DocumentDetail, status_code=201)
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

    # Bridge to prototype in-memory tracking using DB document id
    record = DocumentRecord(id=doc.id, filename=file.filename, status="pending")
    _DOCUMENTS[doc.id] = record

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
    return DocumentDetail(**asdict(record))


@router.get("", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    return [DocumentSummary(**asdict(r)) for r in _DOCUMENTS.values()]


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str) -> DocumentDetail:
    record = _DOCUMENTS.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(**asdict(record))


@router.get("/status/{doc_id}", response_model=DocumentStatus)
async def get_document_status(doc_id: str) -> DocumentStatus:
    record = _DOCUMENTS.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatus(id=record.id, status=record.status)


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


@router.post("/detect-changes", response_model=DetectChangesResponse)
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


@router.put("/apply-changes", response_model=ApplyChangesResponse, status_code=200)
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
