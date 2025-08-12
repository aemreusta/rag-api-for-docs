from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/docs", tags=["Documents"])


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


@router.post("/upload", response_model=DocumentDetail, status_code=201)
async def upload_document(file: UploadFile = UPLOAD_FILE_FORM) -> DocumentDetail:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    doc_id = str(uuid.uuid4())
    record = DocumentRecord(id=doc_id, filename=file.filename, status="pending")
    _DOCUMENTS[doc_id] = record

    # Note: Real implementation will persist file, enqueue job, and return job-aware status
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
