from __future__ import annotations

from dataclasses import dataclass

from fastapi import UploadFile

from app.core.logging_config import get_logger

logger = get_logger(__name__)


ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None


class QualityAssurance:
    """Basic upload validation checks for document ingestion."""

    @staticmethod
    async def validate_upload(file: UploadFile) -> ValidationResult:
        """Validate uploaded file for basic constraints.

        Checks:
        - non-empty filename
        - allowed MIME types (best-effort based on provided content_type)
        - size limit (reads stream in controlled chunks)
        """
        if file is None or not file.filename:
            return ValidationResult(False, "file_missing")

        # MIME type check (best-effort; still re-check server-side when parsing)
        content_type = (file.content_type or "").lower()
        if content_type and content_type not in ALLOWED_MIME_TYPES:
            logger.info(
                "upload_validation_rejected",
                extra={"filename": file.filename, "content_type": content_type},
            )
            return ValidationResult(False, "unsupported_type")

        # Size check: stream-read to avoid loading entire file in memory
        size_read = 0
        chunk_size = 512 * 1024  # 512KB
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            size_read += len(chunk)
            if size_read > MAX_FILE_SIZE_BYTES:
                logger.info(
                    "upload_validation_rejected",
                    extra={"filename": file.filename, "size_bytes": size_read},
                )
                await file.seek(0)
                return ValidationResult(False, "too_large")

        # rewind for downstream consumers
        await file.seek(0)

        logger.info(
            "upload_validation_passed",
            extra={
                "filename": file.filename,
                "content_type": content_type,
                "size_bytes": size_read,
            },
        )
        return ValidationResult(True)
