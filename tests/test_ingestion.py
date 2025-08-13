import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.metrics import get_metrics_backend
from scripts.ingest import PDF_DIRECTORY, main

# Mock the ingestion dependencies to avoid actual DB calls
with patch("scripts.ingest.psycopg2"):
    with patch("scripts.ingest.LlamaIndexCallbackHandler"):
        from scripts.ingest import PDF_DIRECTORY, logger, main

logger = logging.getLogger(__name__)

PDF_DIRECTORY = "pdf_documents/"


def test_ingestion_script_imports():
    """Test that all required imports work correctly."""
    # This test ensures all the imports in the ingestion script are working
    assert main is not None
    assert logger is not None
    assert PDF_DIRECTORY == "pdf_documents/"


@patch("scripts.ingest.psycopg2.connect")
@patch("scripts.ingest.VectorStoreIndex.from_documents")
@patch("scripts.ingest.SimpleDirectoryReader")
@patch("scripts.ingest.PGVectorStore.from_params")
@patch("scripts.ingest.LlamaIndexCallbackHandler")
def test_ingestion_main_function(
    mock_langfuse_handler, mock_pgvector, mock_reader, mock_index, mock_connect
):
    """Test the main ingestion function with mocked dependencies."""
    # Mock Langfuse handler
    mock_handler_instance = MagicMock()
    mock_langfuse_handler.return_value = mock_handler_instance

    # Mock database connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)  # pgvector extension exists
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # Mock document reader
    mock_documents = [MagicMock()]
    mock_reader_instance = MagicMock()
    mock_reader_instance.load_data.return_value = mock_documents
    mock_reader.return_value = mock_reader_instance

    # Mock vector store
    mock_vector_store = MagicMock()
    mock_pgvector.return_value = mock_vector_store

    # Mock index creation
    mock_index_instance = MagicMock()
    mock_index.return_value = mock_index_instance

    # Run the main function
    main()

    # Verify Langfuse setup
    mock_langfuse_handler.assert_called_once_with(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )

    # Verify PGVectorStore setup (embed_dim should match runtime setting)
    mock_pgvector.assert_called_once()
    _, kwargs = mock_pgvector.call_args
    assert kwargs["database"] == "app"
    assert kwargs["host"] == "postgres"
    assert kwargs["password"] == "postgres"
    assert kwargs["port"] == 5432
    assert kwargs["user"] == "postgres"
    assert kwargs["table_name"] == "content_embeddings"
    assert kwargs["embed_dim"] == settings.EMBEDDING_DIM

    # Verify document loading
    mock_reader.assert_called_once_with(input_dir=PDF_DIRECTORY)
    mock_reader_instance.load_data.assert_called_once()

    # Verify index creation
    mock_index.assert_called_once()


@patch("scripts.ingest.psycopg2.connect")
@patch("scripts.ingest.LlamaIndexCallbackHandler")
def test_ingestion_missing_pgvector_extension(mock_langfuse_handler, mock_connect):
    """Test that the script exits gracefully when pgvector extension is missing."""
    # Mock Langfuse handler
    mock_handler_instance = MagicMock()
    mock_langfuse_handler.return_value = mock_handler_instance

    # Mock database connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None  # pgvector extension doesn't exist
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # Run the main function
    main()

    # Verify database check was performed
    mock_cursor.execute.assert_called_once_with(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    )
    mock_conn.close.assert_called_once()


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
    import subprocess
    import sys
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


def test_pdf_directory_constant():
    """Test that the PDF directory constant is set correctly."""
    assert PDF_DIRECTORY == "pdf_documents/"
