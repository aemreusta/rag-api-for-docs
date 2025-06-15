import os
from unittest.mock import MagicMock, patch

from app.core.config import settings
from scripts.ingest import main


def test_ingestion_script_imports():
    """Test that all required imports work correctly."""
    # This test ensures all the imports in the ingestion script are working
    from scripts.ingest import PDF_DIRECTORY, logger, main

    assert main is not None
    assert logger is not None
    assert PDF_DIRECTORY == "pdf_documents/"


@patch("scripts.ingest.psycopg2.connect")
@patch("scripts.ingest.VectorStoreIndex.from_documents")
@patch("scripts.ingest.SimpleDirectoryReader")
@patch("scripts.ingest.PGVectorStore.from_params")
def test_ingestion_main_function(mock_pgvector, mock_reader, mock_index, mock_connect):
    """Test the main ingestion function with mocked dependencies."""
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

    # Verify the function calls
    mock_connect.assert_called_once_with(settings.DATABASE_URL)
    mock_cursor.execute.assert_called_once_with(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    )
    mock_reader.assert_called_once_with(input_dir="pdf_documents/")
    mock_reader_instance.load_data.assert_called_once()
    mock_pgvector.assert_called_once_with(
        dsn=settings.DATABASE_URL,
        table_name="charity_policies",
        embed_dim=384,
    )


@patch("scripts.ingest.psycopg2.connect")
def test_ingestion_missing_pgvector_extension(mock_connect):
    """Test that the script exits gracefully when pgvector extension is missing."""
    # Mock database connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None  # pgvector extension doesn't exist
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # Run the main function
    main()

    # Verify the function exits early
    mock_connect.assert_called_once_with(settings.DATABASE_URL)
    mock_cursor.execute.assert_called_once_with(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    )
    mock_conn.close.assert_called_once()


def test_pdf_directory_exists():
    """Test that the PDF directory exists."""
    assert os.path.exists("pdf_documents/"), "PDF documents directory should exist"


def test_sample_pdf_exists():
    """Test that our sample PDF file exists."""
    sample_pdf = "pdf_documents/sample_policy.pdf"
    assert os.path.exists(sample_pdf), f"Sample PDF should exist at {sample_pdf}"
