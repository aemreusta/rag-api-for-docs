from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import Document, DocumentChunk


def compute_sha256_hex(data: bytes | str) -> str:
    """Compute SHA-256 hex digest for bytes or text input."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ChunkInput:
    index: int
    content: str
    page_number: int | None = None
    section_title: str | None = None
    extra_metadata: dict | None = None


class ContentDeduplicator:
    """Content-based deduplication and versioning helper.

    - Prevents duplicate documents by SHA-256 content hash
    - Bumps version for the same logical document (storage_uri preferred) when content changes
    - Deduplicates chunks within a document by content hash, updating only changed indices
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    # ----------------------------
    # Document-level
    # ----------------------------

    def upsert_document_by_hash(
        self,
        db: Session,
        *,
        filename: str,
        storage_uri: str,
        mime_type: str,
        file_size: int,
        page_count: int | None,
        new_content_hash: str,
        language: str = "en",
        created_by: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[Document, str]:
        """Create or update a document based on content hash and identity.

        Returns (document, action) where action ∈ {"existing", "created", "version_bumped"}.
        Policy:
        - If a document exists with the same content_hash → return (existing)
        - If a document exists with the same storage_uri → bump version and update hash
        - Otherwise create a new document
        """

        # 1) Exact duplicate by hash
        existing_by_hash = db.execute(
            select(Document).where(Document.content_hash == new_content_hash)
        ).scalar_one_or_none()
        if existing_by_hash is not None:
            self._logger.info(
                "document_dedup_hit", document_id=existing_by_hash.id, storage_uri=storage_uri
            )
            return existing_by_hash, "existing"

        # 2) Same logical doc by storage_uri → bump version
        existing_by_uri = db.execute(
            select(Document).where(Document.storage_uri == storage_uri)
        ).scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if existing_by_uri is not None:
            existing_by_uri.version = (existing_by_uri.version or 0) + 1
            existing_by_uri.content_hash = new_content_hash
            existing_by_uri.updated_at = now
            existing_by_uri.page_count = page_count
            existing_by_uri.file_size = file_size
            existing_by_uri.mime_type = mime_type
            if tags:
                # type: ignore[attr-defined]
                existing_by_uri.tags = tags  # ARRAY(String) in model
            db.add(existing_by_uri)
            db.commit()
            self._logger.info(
                "document_version_bumped",
                document_id=existing_by_uri.id,
                version=existing_by_uri.version,
            )
            return existing_by_uri, "version_bumped"

        # 3) New document
        doc = Document(
            id=compute_sha256_hex(storage_uri)[:32],  # deterministic id for dev, replace as needed
            filename=filename,
            original_filename=filename,
            content_hash=new_content_hash,
            file_size=file_size,
            mime_type=mime_type,
            page_count=page_count,
            language=language,
            uploaded_at=now,
            updated_at=now,
            author=created_by,
            storage_backend="local",
            storage_uri=storage_uri,
            version=1,
        )
        db.add(doc)
        db.commit()
        self._logger.info("document_created", document_id=doc.id, storage_uri=storage_uri)
        return doc, "created"

    # ----------------------------
    # Chunk-level
    # ----------------------------

    def upsert_chunks(
        self,
        db: Session,
        *,
        document_id: str,
        chunks: Iterable[ChunkInput],
    ) -> dict[str, int]:
        """Insert new or update changed chunks by index for the specified document.

        Returns counters: {"inserted": n, "updated": m, "unchanged": k}
        """

        # Load existing chunks by index for quick comparison
        existing = db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ).scalars()
        index_to_chunk: dict[int, DocumentChunk] = {c.chunk_index: c for c in existing}

        inserted = updated = unchanged = 0
        for ci in chunks:
            content_hash = compute_sha256_hex(ci.content)
            prior = index_to_chunk.get(ci.index)
            if prior is None:
                # insert new chunk
                new_row = DocumentChunk(
                    id=f"{document_id}:{ci.index}",
                    document_id=document_id,
                    chunk_index=ci.index,
                    content=ci.content,
                    content_hash=content_hash,
                    page_number=ci.page_number,
                    section_title=ci.section_title,
                    extra_metadata=(ci.extra_metadata or {}),
                    created_at=datetime.now(timezone.utc),
                )
                db.add(new_row)
                inserted += 1
            else:
                if prior.content_hash == content_hash:
                    unchanged += 1
                else:
                    # update only when content changed
                    prior.content = ci.content
                    prior.content_hash = content_hash
                    prior.page_number = ci.page_number
                    prior.section_title = ci.section_title
                    prior.extra_metadata = ci.extra_metadata or {}
                    updated += 1

        db.commit()
        return {"inserted": inserted, "updated": updated, "unchanged": unchanged}
