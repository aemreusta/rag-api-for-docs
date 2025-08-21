import logging
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.metrics import get_metrics_backend
from app.main import app

logger = logging.getLogger(__name__)


def test_pdf_directory_exists():
    """Test that the PDF directory exists."""
    assert os.path.exists("pdf_documents/"), "PDF documents directory should exist"


def test_ingest_metrics_backend_smoke():
    metrics = get_metrics_backend()
    # Should not raise when recording ingest metrics regardless of backend
    metrics.increment_counter("ingest_requests_total", {"status": "accepted"})
    metrics.record_histogram("ingest_latency_seconds", 0.01, {"status": "accepted"})


def test_sample_pdfs_and_texts_exist():
    """Ensure generated sample PDFs and text excerpts exist, generating them if missing."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    samples_dir = project_root / "pdf_documents" / "samples"
    expected_files = [
        "hürriyet_partisi_tüzüğü_v3_page1.pdf",
        "hürriyet_partisi_tüzüğü_v3_pages1-3.pdf",
        "hürriyet_partisi_tüzüğü_v3_excerpt_p10-14.pdf",
        "hürriyet_partisi_tüzüğü_v3_full.txt",
        "hürriyet_partisi_tüzüğü_v3_excerpt_pages1-3.txt",
    ]

    samples_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in expected_files if not (samples_dir / f).exists()]
    if missing:
        script = project_root / "scripts" / "generate_samples_from_pdf.py"
        source_pdf = project_root / "pdf_documents" / "Hürriyet Partisi Tüzüğü v3.pdf"
        assert source_pdf.exists(), f"Source policy PDF missing at {source_pdf}"
        try:
            subprocess.run([sys.executable, str(script), str(source_pdf)], check=True)
        except subprocess.CalledProcessError as e:
            raise AssertionError(f"Failed to generate samples via {script}: {e}") from e

    still_missing = [f for f in expected_files if not (samples_dir / f).exists()]
    assert not still_missing, f"Missing generated sample files: {still_missing}"


def test_langfuse_imports():
    """Test that Langfuse imports work correctly."""
    try:
        from llama_index.callbacks.langfuse.base import LlamaIndexCallbackHandler

        assert LlamaIndexCallbackHandler is not None
    except ImportError as e:
        pytest.fail(f"Langfuse imports failed: {e}")


@pytest.mark.asyncio
async def test_settings_available():
    """Test that all required settings are available."""
    assert hasattr(settings, "DATABASE_URL")
    assert hasattr(settings, "LANGFUSE_PUBLIC_KEY")
    assert hasattr(settings, "LANGFUSE_SECRET_KEY")
    assert hasattr(settings, "LANGFUSE_HOST")


def test_ingest_api_list_documents_smoke():
    """Basic smoke test for new ingest API list endpoint."""
    client = TestClient(app)
    r = client.get("/api/v1/docs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
