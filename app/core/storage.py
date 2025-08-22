from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.core.logging_config import get_logger


def compute_content_hash(file_data: bytes) -> str:
    """Compute SHA-256 hash of file data for content-addressed storage."""
    return hashlib.sha256(file_data).hexdigest()


def generate_storage_key(content_hash: str, filename: str) -> str:
    """Generate storage key using content hash and filename.

    Format: <content_hash>/<safe_filename>
    This prevents filename collisions while maintaining content-based deduplication.
    """
    return f"{content_hash}/{filename}"


def parse_storage_key(storage_key: str) -> tuple[str, str]:
    """Parse storage key back into content_hash and filename.

    Returns:
        tuple: (content_hash, filename) or (None, filename) if not content-addressed
    """
    if "/" in storage_key:
        parts = storage_key.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, storage_key


class StorageBackend(Protocol):
    def store_file(self, file_data: bytes, filename: str, content_hash: str | None = None) -> str:
        """Store file and return a storage URI (e.g., file:///... or s3://...).

        Uses content-addressed storage when content_hash is provided:
        - Key format: <content_hash>/<safe_filename>
        - Prevents duplicate storage for identical content
        """

    def retrieve_file(self, uri: str) -> bytes:
        """Retrieve file bytes by storage URI."""


class LocalStorageBackend:
    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger(__name__)

    def store_file(self, file_data: bytes, filename: str, content_hash: str | None = None) -> str:
        # Use content-addressed storage when content_hash is provided
        if content_hash:
            # Create hash-based directory structure: <hash>/<filename>
            hash_dir = self._base / content_hash[:2] / content_hash[2:4]
            hash_dir.mkdir(parents=True, exist_ok=True)
            path = hash_dir / filename

            # Check if file already exists (storage-level deduplication)
            if path.exists():
                # Verify the existing file has the same content
                existing_hash = compute_content_hash(path.read_bytes())
                if existing_hash == content_hash:
                    self._logger.info(
                        "file_dedup_hit", filename=filename, content_hash=content_hash
                    )
                    uri = f"file://{path.resolve()}"
                    return uri
        else:
            # Fallback to filename-based storage for backward compatibility
            path = self._base / filename

        # Store the file
        path.write_bytes(file_data)
        uri = f"file://{path.resolve()}"
        self._logger.info("file_stored", uri=uri, content_hash=content_hash)
        return uri

    def retrieve_file(self, uri: str) -> bytes:
        # Expect file:// URI
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        path = Path(parsed.path)
        return path.read_bytes()


class MinIOStorageBackend:
    def __init__(self):
        # Lazy import to avoid hard dependency in dev
        try:
            from minio import Minio  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("minio client is not installed") from e

        secure = bool(settings.MINIO_SECURE)
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=secure,
        )
        self._bucket = settings.MINIO_BUCKET
        self._logger = get_logger(__name__)

        # Ensure bucket exists
        if not self._client.bucket_exists(self._bucket):  # pragma: no cover
            self._client.make_bucket(self._bucket)

    def store_file(self, file_data: bytes, filename: str, content_hash: str | None = None) -> str:
        # Use content-addressed storage when content_hash is provided
        if content_hash:
            # Create hash-based key: <content_hash>/<safe_filename>
            key = f"{content_hash}/{filename}"

            # Check if object already exists (storage-level deduplication)
            try:
                stat = self._client.stat_object(self._bucket, key)  # pragma: no cover
                if stat.size == len(file_data):
                    # Object exists with same size, likely same content
                    self._logger.info(
                        "file_dedup_hit", filename=filename, content_hash=content_hash, key=key
                    )
                    uri = f"s3://{self._bucket}/{key}"
                    return uri
            except Exception:
                # Object doesn't exist, proceed with upload
                pass
        else:
            # Fallback to filename-based storage for backward compatibility
            key = filename

        # Store the file
        from io import BytesIO

        self._client.put_object(
            self._bucket, key, BytesIO(file_data), length=len(file_data)
        )  # pragma: no cover
        uri = f"s3://{self._bucket}/{key}"
        self._logger.info("file_stored", uri=uri, content_hash=content_hash)
        return uri

    def retrieve_file(self, uri: str) -> bytes:
        # s3://bucket/key
        from io import BytesIO
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        bucket = parsed.netloc or self._bucket
        key = parsed.path.lstrip("/")
        data = BytesIO()
        self._client.get_object(bucket, key).readinto(data)  # pragma: no cover
        return data.getvalue()


def get_storage_backend() -> StorageBackend:
    if getattr(settings, "STORAGE_BACKEND", "local").lower() == "minio":  # pragma: no cover
        return MinIOStorageBackend()
    return LocalStorageBackend(settings.UPLOADED_DOCS_DIR)
