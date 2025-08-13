from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_and_list_documents():
    # Upload a fake file (mock storage backend to avoid filesystem writes)
    with patch("app.api.v1.docs.storage") as mock_storage:
        mock_storage.store_file.return_value = "file:///tmp/uploaded_docs/sample.txt"
        files = {"file": ("sample.txt", b"hello-unique", "text/plain")}
        r = client.post("/api/v1/docs/upload", files=files)
    assert r.status_code == 201
    doc = r.json()
    assert "id" in doc and doc["filename"] == "sample.txt" and doc["status"] == "pending"

    # List should contain the uploaded document
    r = client.get("/api/v1/docs")
    assert r.status_code == 200
    items = r.json()
    assert any(item["id"] == doc["id"] for item in items)

    # Re-upload same content should still succeed; storage may be skipped via cache
    with patch("app.api.v1.docs.storage") as mock_storage2:
        mock_storage2.store_file.return_value = "file:///tmp/uploaded_docs/sample.txt"
        files = {"file": ("sample.txt", b"hello", "text/plain")}
        r2 = client.post("/api/v1/docs/upload", files=files)
        assert r2.status_code == 201


def test_upload_calls_deduplicator():
    class FakeCache:
        def __init__(self):
            self._store = {}

        async def get(self, key: str):
            return self._store.get(key)

        async def set(self, key: str, value, ttl: int = 600):
            self._store[key] = value

    with (
        patch("app.api.v1.docs.get_cache_backend", return_value=FakeCache()),
        patch("app.api.v1.docs.storage") as mock_storage,
        patch("app.api.v1.docs.ContentDeduplicator.upsert_document_by_hash") as mock_upsert,
    ):
        mock_storage.store_file.return_value = "file:///tmp/uploaded_docs/sample.txt"
        mock_upsert.return_value = (SimpleNamespace(id="doc-abc"), "created")

        files = {"file": ("sample.txt", b"hello", "text/plain")}
        r = client.post("/api/v1/docs/upload", files=files)
        assert r.status_code == 201

        # storage used and deduplicator called with expected args
        mock_storage.store_file.assert_called_once()
        _, kwargs = mock_upsert.call_args
        assert kwargs["filename"] == "sample.txt"
        assert kwargs["storage_uri"].startswith("file://")
        assert kwargs["mime_type"] == "text/plain"
        assert kwargs["file_size"] == len(b"hello")
        assert kwargs["page_count"] is None


@pytest.mark.minio
def test_upload_uses_minio_storage():
    class DummyMinio:
        def store_file(self, file_data: bytes, filename: str) -> str:  # noqa: D401
            return f"s3://uploads/{filename}"

    with (
        patch("app.api.v1.docs.storage", DummyMinio()),
        patch("app.api.v1.docs.ContentDeduplicator.upsert_document_by_hash") as mock_upsert,
    ):
        mock_upsert.return_value = (SimpleNamespace(id="doc-minio"), "created")
        files = {"file": ("minio.txt", b"data", "text/plain")}
        r = client.post("/api/v1/docs/upload", files=files)
        assert r.status_code == 201

        # Ensure dedup received s3 URI from MinIO-like backend
        _, kwargs = mock_upsert.call_args
        assert kwargs["storage_uri"].startswith("s3://")


def test_upload_uses_storage_cache_by_content_hash():
    class FakeCache:
        def __init__(self):
            self._store = {}

        async def get(self, key: str):  # noqa: D401
            return self._store.get(key)

        async def set(self, key: str, value, ttl: int = 600):  # noqa: D401
            self._store[key] = value

    fake_cache = FakeCache()

    with (
        patch("app.api.v1.docs.get_cache_backend", return_value=fake_cache),
        patch("app.api.v1.docs.storage") as mock_storage,
    ):
        mock_storage.store_file.return_value = "file:///tmp/uploaded_docs/sample.txt"
        files = {"file": ("cached.txt", b"same-bytes", "text/plain")}
        # First upload should call storage and populate cache
        r1 = client.post("/api/v1/docs/upload", files=files)
        assert r1.status_code == 201
        assert mock_storage.store_file.call_count == 1

        # Second upload with same content should hit cache and not call storage again
        r2 = client.post("/api/v1/docs/upload", files=files)
        assert r2.status_code == 201
        assert mock_storage.store_file.call_count == 1


def test_upload_validation_rejects_large_file():
    # Create a payload just over 10MB
    too_big = b"x" * (10 * 1024 * 1024 + 1)
    files = {"file": ("big.pdf", too_big, "application/pdf")}
    r = client.post("/api/v1/docs/upload", files=files)
    assert r.status_code == 400
    assert "Invalid upload" in r.text


def test_upload_validation_rejects_unsupported_type():
    files = {"file": ("image.png", b"fake-bytes", "image/png")}
    r = client.post("/api/v1/docs/upload", files=files)
    assert r.status_code == 400
    assert "Invalid upload" in r.text


def test_get_document_and_status():
    # Upload first
    files = {"file": ("readme.md", b"content", "text/markdown")}
    r = client.post("/api/v1/docs/upload", files=files)
    assert r.status_code == 201
    doc = r.json()

    # Get document
    r = client.get(f"/api/v1/docs/{doc['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == doc["id"] and body["filename"] == "readme.md"

    # Status
    r = client.get(f"/api/v1/docs/status/{doc['id']}")
    assert r.status_code == 200
    status = r.json()
    assert status["id"] == doc["id"] and status["status"] == "pending"


def test_scrape_url_accepted():
    r = client.post("/api/v1/docs/scrape", json={"url": "https://example.com"})
    assert r.status_code == 202
    body = r.json()
    assert body["url"] == "https://example.com" and body["status"] == "pending"
