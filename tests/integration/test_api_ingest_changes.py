from __future__ import annotations

import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.deps import get_db_session
from app.db.models import Document
from app.main import app

client = TestClient(app)


def test_detect_changes_document_not_found():
    payload = {
        "document_id": "non-existent",
        "new_content_hash": "abc123",
        "page_hashes": None,
    }

    r = client.post("/api/v1/docs/detect-changes", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["is_new_version"] is True
    assert body["changed_pages"] == []
    assert body["reason"] == "document_not_found"


def test_jobs_status_shape():
    class DummyAsyncResult:
        def __init__(self, state: str, info: dict | None = None):
            self.state = state
            self.info = info

    job_id = str(uuid.uuid4())
    with patch("app.api.v1.docs.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = DummyAsyncResult("PENDING")
        r = client.get(f"/api/v1/docs/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"id", "status", "progress_percent", "detail"}


def test_apply_changes_with_existing_document(db_session):
    # Override dependency to use test DB session
    app.dependency_overrides[get_db_session] = lambda: db_session

    try:
        # Create a minimal document row
        doc = Document(
            id=str(uuid.uuid4()),
            filename="doc.pdf",
            original_filename="doc.pdf",
            content_hash="oldhash",
            file_size=10,
            mime_type="application/pdf",
            page_count=2,
            version=1,
            storage_uri="file:///tmp/doc.pdf",
        )
        db_session.add(doc)
        db_session.commit()

        # Prepare payload: mark page 1 as changed
        payload = {
            "document_id": doc.id,
            "changes": {
                "is_new_version": True,
                "changed_pages": [1],
                "reason": "hash_changed",
            },
            "new_content_hash": "newhash",
            "page_texts": {"pages": {1: "updated page one text"}},
        }

        # Monkeypatch heavy processing to no-op for API contract
        import app.core.incremental as inc

        orig = inc.IncrementalProcessor.process_incremental_update
        inc.IncrementalProcessor.process_incremental_update = (  # type: ignore[assignment]
            lambda *args, **kwargs: None
        )
        try:
            r = client.put("/api/v1/docs/apply-changes", json=payload)
        finally:
            inc.IncrementalProcessor.process_incremental_update = orig  # type: ignore[assignment]

        assert r.status_code == 200
        body = r.json()
        assert body["updated_pages"] == [1]
        # version may be None or unchanged since we patched processing; assert presence
        assert "version" in body
    finally:
        app.dependency_overrides.clear()
