from __future__ import annotations

import socket
import uuid

import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.incremental import ChangeSet, IncrementalProcessor
from app.db.models import ContentEmbedding, Document


def _db_host_resolves() -> bool:
    try:
        socket.getaddrinfo(settings.POSTGRES_SERVER, 5432)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_host_resolves(), reason="Postgres host not resolvable in this environment"
)


def test_detect_changes_no_change(db_session: Session):
    doc = Document(
        id=str(uuid.uuid4()),
        filename="sample.pdf",
        content_hash="abc",
        page_count=3,
        version=1,
        storage_uri="file:///tmp/sample.pdf",
        mime_type="application/pdf",
        file_size=10,
    )
    db_session.add(doc)
    db_session.commit()

    proc = IncrementalProcessor()
    changes = proc.detect_changes(db=db_session, document_id=doc.id, new_content_hash="abc")
    assert changes.changed_pages == [] and changes.reason == "no_change"


def test_detect_changes_full_when_no_page_hashes(db_session: Session):
    doc = Document(
        id=str(uuid.uuid4()),
        filename="sample.pdf",
        content_hash="old",
        page_count=2,
        version=1,
        storage_uri="file:///tmp/sample.pdf",
        mime_type="application/pdf",
        file_size=10,
    )
    db_session.add(doc)
    db_session.commit()

    proc = IncrementalProcessor()
    changes = proc.detect_changes(db=db_session, document_id=doc.id, new_content_hash="new")
    assert changes.changed_pages == [1, 2]


@pytest.mark.integration
def test_process_incremental_update_reembeds_selected_pages(db_session: Session, monkeypatch):
    doc = Document(
        id=str(uuid.uuid4()),
        filename="doc.pdf",
        content_hash="old",
        page_count=3,
        version=1,
        storage_uri="file:///tmp/doc.pdf",
        mime_type="application/pdf",
        file_size=10,
    )
    db_session.add(doc)
    db_session.commit()

    # Seed an existing embedding row for page 2 to be replaced
    db_session.add(
        ContentEmbedding(source_document=doc.filename, page_number=2, content_text="old text")
    )
    db_session.commit()

    # Avoid external embedding during unit test by stubbing out re-embedding
    calls: list[dict] = []

    def _noop_reembed(self, *, page_texts, source_document):  # type: ignore[no-redef]
        calls.append({"page_texts": dict(page_texts), "source_document": source_document})

    monkeypatch.setattr("app.core.incremental.IncrementalProcessor._reembed_pages", _noop_reembed)

    proc = IncrementalProcessor()
    change_set = ChangeSet(is_new_version=True, changed_pages=[2], reason="hash_changed")

    page_texts = {2: "This page changed. It now has updated content."}

    proc.process_incremental_update(
        db=db_session,
        document_id=doc.id,
        changes=change_set,
        new_content_hash="new",
        new_page_texts=page_texts,
    )

    updated = db_session.get(Document, doc.id)
    assert updated is not None
    assert updated.version == 2
    assert updated.content_hash == "new"

    # Existing embedding row for page 2 should be deleted
    remaining = (
        db_session.query(ContentEmbedding)
        .filter(ContentEmbedding.source_document == doc.filename)
        .count()
    )
    assert remaining == 0

    # Ensure our stub was invoked with expected texts
    assert calls and 2 in calls[0]["page_texts"]
