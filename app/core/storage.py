from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.core.logging_config import get_logger


class StorageBackend(Protocol):
    def store_file(self, file_data: bytes, filename: str) -> str:
        """Store file and return a storage URI (e.g., file:///... or s3://...)."""

    def retrieve_file(self, uri: str) -> bytes:
        """Retrieve file bytes by storage URI."""


class LocalStorageBackend:
    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger(__name__)

    def store_file(self, file_data: bytes, filename: str) -> str:
        # Caller provides sanitized filename
        path = self._base / filename
        path.write_bytes(file_data)
        uri = f"file://{path.resolve()}"
        self._logger.info("file_stored", uri=uri)
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

    def store_file(self, file_data: bytes, filename: str) -> str:
        # Simple key under optional prefix
        key = filename
        from io import BytesIO

        self._client.put_object(
            self._bucket, key, BytesIO(file_data), length=len(file_data)
        )  # pragma: no cover
        uri = f"s3://{self._bucket}/{key}"
        self._logger.info("file_stored", uri=uri)
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
