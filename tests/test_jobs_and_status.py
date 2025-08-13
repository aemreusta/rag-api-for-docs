from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_enqueues_background_job():
    with (
        patch("app.api.v1.docs.storage") as mock_storage,
        patch("app.api.v1.docs.ContentDeduplicator.upsert_document_by_hash") as mock_upsert,
        patch("app.api.v1.docs.process_document_async.delay") as mock_delay,
    ):
        mock_storage.store_file.return_value = "file:///tmp/uploaded_docs/sample.txt"
        mock_upsert.return_value = (SimpleNamespace(id="doc-enqueue"), "created")

        files = {"file": ("sample.txt", b"hello", "text/plain")}
        r = client.post("/api/v1/docs/upload", files=files)
        assert r.status_code == 201
        # Ensure enqueued
        assert mock_delay.called


def test_job_status_endpoint_maps_states():
    class DummyAsyncResult:
        def __init__(self, state: str, info: dict | None = None):
            self.state = state
            self.info = info

    with patch("app.api.v1.docs.celery_app") as mock_celery:
        mock_celery.AsyncResult.side_effect = [
            DummyAsyncResult("PENDING"),
            DummyAsyncResult("STARTED"),
            DummyAsyncResult("RETRY"),
            DummyAsyncResult("FAILURE", {"error": "boom"}),
            DummyAsyncResult("SUCCESS", {"ok": True}),
        ]

        assert client.get("/api/v1/docs/jobs/a").json()["status"] == "pending"
        assert client.get("/api/v1/docs/jobs/b").json()["status"] == "processing"
        assert client.get("/api/v1/docs/jobs/c").json()["status"] == "retry"
        assert client.get("/api/v1/docs/jobs/d").json()["status"] == "failed"
        assert client.get("/api/v1/docs/jobs/e").json()["status"] == "completed"
