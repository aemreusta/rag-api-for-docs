from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_and_list_documents():
    # Upload a fake file
    files = {"file": ("sample.txt", b"hello", "text/plain")}
    r = client.post("/api/v1/docs/upload", files=files)
    assert r.status_code == 201
    doc = r.json()
    assert "id" in doc and doc["filename"] == "sample.txt" and doc["status"] == "pending"

    # List should contain the uploaded document
    r = client.get("/api/v1/docs")
    assert r.status_code == 200
    items = r.json()
    assert any(item["id"] == doc["id"] for item in items)


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
