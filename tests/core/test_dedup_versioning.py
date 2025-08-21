from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.core.dedup import ChunkInput, ContentDeduplicator, compute_sha256_hex
from app.db.models import Document, DocumentChunk


def test_document_dedup_and_version_bump(db_session: Session):
    dedup = ContentDeduplicator()
    storage_uri = "file:///tmp/sample.pdf"

    # First insert
    doc, action = dedup.upsert_document_by_hash(
        db_session,
        filename="sample.pdf",
        storage_uri=storage_uri,
        mime_type="application/pdf",
        file_size=123,
        page_count=2,
        new_content_hash=compute_sha256_hex("v1"),
    )
    assert action == "created"
    v1_id = doc.id
    assert doc.version == 1

    # Exact duplicate by hash → existing
    doc2, action2 = dedup.upsert_document_by_hash(
        db_session,
        filename="sample.pdf",
        storage_uri=storage_uri,
        mime_type="application/pdf",
        file_size=123,
        page_count=2,
        new_content_hash=compute_sha256_hex("v1"),
    )
    assert action2 == "existing"
    assert doc2.id == v1_id

    # Change content but same logical doc (same storage_uri) → bump version
    doc3, action3 = dedup.upsert_document_by_hash(
        db_session,
        filename="sample.pdf",
        storage_uri=storage_uri,
        mime_type="application/pdf",
        file_size=124,
        page_count=3,
        new_content_hash=compute_sha256_hex("v2"),
    )
    assert action3 == "version_bumped"
    assert doc3.id == v1_id
    assert doc3.version == 2


def test_chunk_upsert_insert_update_unchanged(db_session: Session):
    # Seed a document
    doc = Document(
        id=str(uuid.uuid4()),
        filename="doc.pdf",
        content_hash=compute_sha256_hex("doc"),
        file_size=10,
        mime_type="application/pdf",
        page_count=2,
        storage_uri="file:///tmp/doc.pdf",
        version=1,
    )
    db_session.add(doc)
    db_session.commit()

    dedup = ContentDeduplicator()

    # First upsert: two inserts
    res1 = dedup.upsert_chunks(
        db_session,
        document_id=doc.id,
        chunks=[
            ChunkInput(index=0, content="hello", page_number=1),
            ChunkInput(index=1, content="world", page_number=1),
        ],
    )
    assert res1 == {"inserted": 2, "updated": 0, "unchanged": 0}

    # Second upsert: one unchanged, one updated
    res2 = dedup.upsert_chunks(
        db_session,
        document_id=doc.id,
        chunks=[
            ChunkInput(index=0, content="hello", page_number=1),
            ChunkInput(index=1, content="WORLD", page_number=1),
        ],
    )
    assert res2 == {"inserted": 0, "updated": 1, "unchanged": 1}

    # Verify database state matches expectations
    rows = (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    assert len(rows) == 2
    assert rows[0].content == "hello"
    assert rows[1].content == "WORLD"
