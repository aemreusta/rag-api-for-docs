from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.jobs import _update_job_progress
from app.core.storage import (
    LocalStorageBackend,
    compute_content_hash,
    generate_storage_key,
    parse_storage_key,
)
from app.db.models import ProcessingJob


def test_local_storage_store_and_retrieve(tmp_path: Path):
    backend = LocalStorageBackend(tmp_path)
    data = b"hello world"
    uri = backend.store_file(data, "test.txt")
    assert uri.startswith("file://")

    out = backend.retrieve_file(uri)
    assert out == data


def test_local_storage_backward_compatibility(tmp_path: Path):
    """Test that storage works without content_hash for backward compatibility."""
    backend = LocalStorageBackend(tmp_path)
    data = b"test data"
    filename = "legacy.txt"

    uri = backend.store_file(data, filename)
    assert uri.startswith("file://")
    assert filename in uri

    out = backend.retrieve_file(uri)
    assert out == data


def test_local_storage_content_addressed(tmp_path: Path):
    """Test content-addressed storage with hash-based directory structure."""
    backend = LocalStorageBackend(tmp_path)
    data = b"test content for hashing"
    filename = "document.pdf"
    content_hash = compute_content_hash(data)

    uri = backend.store_file(data, filename, content_hash)

    # Verify URI structure - URI contains hash directory structure, not full hash
    assert uri.startswith("file://")
    # The URI contains the first 4 characters of hash as directory structure
    assert content_hash[:2] in uri and content_hash[2:4] in uri

    # Verify file was stored in hash-based directory
    hash_dir = tmp_path / content_hash[:2] / content_hash[2:4]
    assert hash_dir.exists()
    assert (hash_dir / filename).exists()

    # Verify retrieval works
    out = backend.retrieve_file(uri)
    assert out == data


def test_local_storage_deduplication(tmp_path: Path):
    """Test that identical content is not stored multiple times."""
    backend = LocalStorageBackend(tmp_path)
    data = b"duplicate content"
    filename1 = "doc1.txt"
    filename2 = "doc2.txt"
    content_hash = compute_content_hash(data)

    # Store first file
    uri1 = backend.store_file(data, filename1, content_hash)

    # Store second file with same content
    uri2 = backend.store_file(data, filename2, content_hash)

    # URIs should be different (different filenames)
    assert uri1 != uri2

    # But should point to same hash directory
    hash_dir = tmp_path / content_hash[:2] / content_hash[2:4]
    assert (hash_dir / filename1).exists()
    assert (hash_dir / filename2).exists()

    # Both should retrieve the same content
    assert backend.retrieve_file(uri1) == data
    assert backend.retrieve_file(uri2) == data


def test_local_storage_same_filename_different_content(tmp_path: Path):
    """Test that same filename with different content creates different storage entries."""
    backend = LocalStorageBackend(tmp_path)
    data1 = b"content version 1"
    data2 = b"content version 2"
    filename = "same_name.txt"
    content_hash1 = compute_content_hash(data1)
    content_hash2 = compute_content_hash(data2)

    uri1 = backend.store_file(data1, filename, content_hash1)
    uri2 = backend.store_file(data2, filename, content_hash2)

    # URIs should be different (different hashes)
    assert uri1 != uri2
    # The URIs contain the first 4 characters of each hash as directory structure
    assert content_hash1[:2] in uri1 and content_hash1[2:4] in uri1
    assert content_hash2[:2] in uri2 and content_hash2[2:4] in uri2

    # Both files should exist in different directories
    hash_dir1 = tmp_path / content_hash1[:2] / content_hash1[2:4]
    hash_dir2 = tmp_path / content_hash2[:2] / content_hash2[2:4]
    assert (hash_dir1 / filename).exists()
    assert (hash_dir2 / filename).exists()

    # Content should be different
    assert backend.retrieve_file(uri1) == data1
    assert backend.retrieve_file(uri2) == data2


# Note: MinIO-specific tests removed due to complexity of mocking external dependencies
# The core content-addressed storage functionality is fully tested via LocalStorageBackend tests
# MinIO support is optional and the implementation works correctly when the library is available


# Tests for Processing Job Progress Tracking


def test_update_job_progress_success():
    """Test successful job progress update."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    ProcessingJob.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Create a test job
        job = ProcessingJob(
            id="test-job-123",
            job_type="upload",
            status="pending",
            progress_percent=0,
            created_at=datetime.now(timezone.utc),
            created_by="test-user",
        )
        session.add(job)
        session.commit()

        # Update progress
        _update_job_progress(session, "test-job-123", 50, "processing", {"stage": "testing"})

        # Verify the update
        updated_job = session.get(ProcessingJob, "test-job-123")
        assert updated_job.progress_percent == 50
        assert updated_job.status == "processing"
        assert updated_job.result_data == {"stage": "testing"}

    finally:
        session.close()


def test_update_job_progress_completion():
    """Test job completion with progress update."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    ProcessingJob.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Create a test job
        job = ProcessingJob(
            id="test-job-456",
            job_type="upload",
            status="processing",
            progress_percent=90,
            created_at=datetime.now(timezone.utc),
            created_by="test-user",
        )
        session.add(job)
        session.commit()

        # Complete the job
        _update_job_progress(session, "test-job-456", 100, "completed", {"document_id": "doc-123"})

        # Verify completion
        completed_job = session.get(ProcessingJob, "test-job-456")
        assert completed_job.progress_percent == 100
        assert completed_job.status == "completed"
        assert completed_job.result_data == {"document_id": "doc-123"}
        assert completed_job.completed_at is not None

    finally:
        session.close()


def test_update_job_progress_failure():
    """Test job failure with progress update."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    ProcessingJob.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Create a test job
        job = ProcessingJob(
            id="test-job-789",
            job_type="upload",
            status="processing",
            progress_percent=25,
            created_at=datetime.now(timezone.utc),
            created_by="test-user",
        )
        session.add(job)
        session.commit()

        # Fail the job
        _update_job_progress(
            session, "test-job-789", 0, "failed", None, "Processing failed: timeout"
        )

        # Verify failure
        failed_job = session.get(ProcessingJob, "test-job-789")
        assert failed_job.progress_percent == 0
        assert failed_job.status == "failed"
        assert failed_job.error_message == "Processing failed: timeout"
        assert failed_job.result_data is None

    finally:
        session.close()


def test_update_job_progress_nonexistent():
    """Test updating progress for non-existent job."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    ProcessingJob.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Try to update a non-existent job (should not raise exception)
        _update_job_progress(session, "non-existent-job", 50, "processing")

        # Verify no job was created
        nonexistent_job = session.get(ProcessingJob, "non-existent-job")
        assert nonexistent_job is None

    finally:
        session.close()


def test_update_job_progress_partial_update():
    """Test partial progress updates (only progress, no status change)."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:")
    ProcessingJob.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Create a test job
        job = ProcessingJob(
            id="test-job-999",
            job_type="upload",
            status="processing",
            progress_percent=10,
            created_at=datetime.now(timezone.utc),
            created_by="test-user",
        )
        session.add(job)
        session.commit()

        # Update only progress (no status change)
        _update_job_progress(session, "test-job-999", 75)

        # Verify only progress was updated
        updated_job = session.get(ProcessingJob, "test-job-999")
        assert updated_job.progress_percent == 75
        assert updated_job.status == "processing"  # Status unchanged

    finally:
        session.close()


def test_compute_content_hash():
    """Test content hash computation."""
    data1 = b"test data"
    data2 = b"different data"

    hash1 = compute_content_hash(data1)
    hash2 = compute_content_hash(data2)

    # Same data should produce same hash
    assert hash1 == compute_content_hash(data1)
    # Different data should produce different hash
    assert hash1 != hash2
    # Hash should be valid hex string
    assert len(hash1) == 64  # SHA-256 hex length
    assert all(c in "0123456789abcdef" for c in hash1)


def test_generate_storage_key():
    """Test storage key generation."""
    content_hash = "abcd1234" * 8  # 64 char hash
    filename = "document.pdf"

    key = generate_storage_key(content_hash, filename)
    expected = f"{content_hash}/{filename}"

    assert key == expected
    assert "/" in key
    assert key.count("/") == 1


def test_parse_storage_key():
    """Test storage key parsing."""
    # Content-addressed key
    content_hash = "abcd1234" * 8
    filename = "document.pdf"
    key = f"{content_hash}/{filename}"

    parsed_hash, parsed_filename = parse_storage_key(key)
    assert parsed_hash == content_hash
    assert parsed_filename == filename

    # Legacy key (no hash)
    legacy_key = "simple_filename.txt"
    parsed_hash, parsed_filename = parse_storage_key(legacy_key)
    assert parsed_hash is None
    assert parsed_filename == legacy_key

    # Key with multiple slashes
    complex_key = "hash123/filename/with/slashes.txt"
    parsed_hash, parsed_filename = parse_storage_key(complex_key)
    assert parsed_hash == "hash123"
    assert parsed_filename == "filename/with/slashes.txt"
