from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite

try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

from fastapi import UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


def _detect_mime_by_extension(filename: str) -> str | None:
    """Detect MIME type based on file extension (fallback when magic unavailable)."""
    extension_map = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
    }

    for ext, mime_type in extension_map.items():
        if filename.lower().endswith(ext):
            return mime_type

    return None


def _is_mime_compatible(declared: str, detected: str) -> bool:
    """Check if declared and detected MIME types are compatible.

    Some MIME type variations are acceptable (e.g., text/plain vs text/markdown).
    """
    # Exact match
    if declared == detected:
        return True

    # Known compatible combinations
    compatible_pairs = {
        ("text/plain", "text/markdown"),
        ("text/markdown", "text/plain"),
        # Add more as needed
    }

    return (declared, detected) in compatible_pairs


def _validate_filename(file: UploadFile) -> ValidationResult:
    """Validate file presence and filename safety."""
    if file is None or not file.filename:
        return ValidationResult(False, "file_missing")

    filename = file.filename.strip()
    if not filename or len(filename) > 255 or ".." in filename:
        logger.warning(
            "upload_validation_rejected",
            extra={"filename": file.filename, "reason": "invalid_filename"},
        )
        return ValidationResult(False, "invalid_filename")

    return ValidationResult(True)


async def _read_and_validate_content(file: UploadFile, filename: str) -> ValidationResult:
    """Read file content and validate size constraints."""
    content_chunks = []
    size_read = 0
    chunk_size = 512 * 1024  # 512KB
    hash_obj = hashlib.sha256()

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        content_chunks.append(chunk)
        hash_obj.update(chunk)
        size_read += len(chunk)

        # Early size rejection
        if size_read > MAX_FILE_SIZE_BYTES:
            logger.info(
                "upload_validation_rejected",
                extra={"filename": filename, "size_bytes": size_read, "reason": "too_large"},
            )
            await file.seek(0)
            return ValidationResult(False, "too_large", file_size_bytes=size_read)

    content_hash = hash_obj.hexdigest()

    # Minimum size check (empty files)
    if size_read == 0:
        logger.info(
            "upload_validation_rejected",
            extra={"filename": filename, "reason": "empty_file"},
        )
        await file.seek(0)
        return ValidationResult(False, "empty_file", content_hash=content_hash)

    # Return content info in ValidationResult fields
    return ValidationResult(
        True,
        reason=content_chunks,  # Use reason field to pass content_chunks
        content_hash=content_hash,
        file_size_bytes=size_read,
    )


def _validate_mime_types(
    file: UploadFile, filename: str, content_chunks: list, content_hash: str, size_read: int
) -> ValidationResult:
    """Validate MIME types with strict enforcement."""
    declared_mime = (file.content_type or "").lower().strip()

    # Detect actual MIME type from content (if magic library available)
    detected_mime = None
    if MAGIC_AVAILABLE:
        full_content = b"".join(content_chunks)
        try:
            detected_mime = magic.from_buffer(full_content, mime=True).lower()
        except Exception as e:
            logger.warning("mime_detection_failed", extra={"filename": filename, "error": str(e)})
            detected_mime = None
    else:
        # Fallback: basic file extension to MIME type mapping
        detected_mime = _detect_mime_by_extension(filename)

    # Strict MIME type enforcement
    if declared_mime and declared_mime not in ALLOWED_MIME_TYPES:
        logger.info(
            "upload_validation_rejected",
            extra={
                "filename": filename,
                "declared_mime": declared_mime,
                "reason": "unsupported_declared_type",
            },
        )
        return ValidationResult(
            False,
            "unsupported_declared_type",
            content_hash=content_hash,
            detected_mime_type=detected_mime,
            file_size_bytes=size_read,
        )

    if detected_mime and detected_mime not in ALLOWED_MIME_TYPES:
        logger.warning(
            "upload_validation_rejected",
            extra={
                "filename": filename,
                "declared_mime": declared_mime,
                "detected_mime": detected_mime,
                "reason": "unsupported_detected_type",
            },
        )
        return ValidationResult(
            False,
            "unsupported_detected_type",
            content_hash=content_hash,
            detected_mime_type=detected_mime,
            file_size_bytes=size_read,
        )

    # MIME type mismatch detection (security)
    if (
        declared_mime
        and detected_mime
        and declared_mime != detected_mime
        and not _is_mime_compatible(declared_mime, detected_mime)
    ):
        logger.warning(
            "upload_validation_rejected",
            extra={
                "filename": filename,
                "declared_mime": declared_mime,
                "detected_mime": detected_mime,
                "reason": "mime_mismatch",
            },
        )
        return ValidationResult(
            False,
            "mime_mismatch",
            content_hash=content_hash,
            detected_mime_type=detected_mime,
            file_size_bytes=size_read,
        )

    return ValidationResult(True, detected_mime_type=detected_mime)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    content_hash: str | None = None
    detected_mime_type: str | None = None
    file_size_bytes: int | None = None


@dataclass(frozen=True)
class QualityScore:
    score: float
    length_chars: int
    length_words: int
    avg_sentence_length: float
    flags: list[str]


@dataclass(frozen=True)
class IntegrityReport:
    ok: bool
    total_rows: int
    null_vectors: int
    wrong_dims: int | None
    hnsw_index_present: bool | None
    notes: list[str]


class QualityAssurance:
    """Quality checks for uploads, content, embeddings, and vector store integrity."""

    @staticmethod
    async def validate_upload(file: UploadFile) -> ValidationResult:
        """Validate uploaded file with comprehensive checks and early checksum computation."""
        # Basic file presence and filename validation
        filename_result = _validate_filename(file)
        if not filename_result.ok:
            return filename_result

        filename = file.filename.strip()

        # Read and validate file content
        content_result = await _read_and_validate_content(file, filename)
        if not content_result.ok:
            return content_result

        content_hash, size_read, content_chunks = (
            content_result.content_hash,
            content_result.file_size_bytes,
            content_result.reason,
        )

        # MIME type validation
        mime_result = _validate_mime_types(file, filename, content_chunks, content_hash, size_read)
        if not mime_result.ok:
            return mime_result

        # Success - rewind file for downstream use
        await file.seek(0)
        logger.info(
            "upload_validation_passed",
            extra={
                "filename": filename,
                "declared_mime": (file.content_type or "").lower().strip(),
                "detected_mime": mime_result.detected_mime_type,
                "size_bytes": size_read,
                "content_hash": content_hash[:12] + "...",
            },
        )

        return ValidationResult(
            True,
            content_hash=content_hash,
            detected_mime_type=mime_result.detected_mime_type,
            file_size_bytes=size_read,
        )

    @staticmethod
    def score_content_quality(content: str) -> QualityScore:
        """Compute a simple quality score for text content (0.0 - 1.0).

        Heuristics (conservative, language-agnostic):
        - Minimum length thresholds
        - Sentence segmentation by '.', '!', '?'
        - Penalize extremely long sentences and excessive repetition
        - Reward balanced sentence lengths and presence of punctuation
        """
        if not content:
            return QualityScore(0.0, 0, 0, 0.0, ["empty"])

        text = " ".join(content.split())
        length_chars = len(text)
        words = text.split()
        length_words = len(words)
        # Naive sentence split
        import re

        sentences = [s.strip() for s in re.split(r"[.!?]+\s*", text) if s.strip()]
        avg_sentence_length = (length_words / len(sentences)) if sentences else float(length_words)

        flags: list[str] = []
        score = 1.0

        # Length checks
        if length_words < 20:
            score -= 0.4
            flags.append("too_short")
        if length_chars > 200_000:
            score -= 0.2
            flags.append("too_long")

        # Sentence length penalty (very long average sentence)
        if avg_sentence_length > 40:
            score -= 0.2
            flags.append("long_sentences")

        # Repetition heuristic: top token frequency ratio
        from collections import Counter

        token_counts = Counter([w.lower() for w in words])
        if token_counts:
            most_common_count = token_counts.most_common(1)[0][1]
            ratio = most_common_count / max(1, length_words)
            if ratio > 0.2:
                score -= 0.2
                flags.append("repetitive")

        # Bonus for having basic punctuation diversity
        punct_bonus = 0.0
        for sym in [",", ";", ":", "(", ")"]:
            if sym in text:
                punct_bonus = 0.05
                break
        score += punct_bonus

        # Clamp score
        score = max(0.0, min(1.0, score))

        return QualityScore(
            score=round(score, 3),
            length_chars=length_chars,
            length_words=length_words,
            avg_sentence_length=round(avg_sentence_length, 2),
            flags=flags,
        )

    @staticmethod
    def validate_embeddings(embeddings: list[float] | list[list[float]]) -> bool:
        """Validate embedding vector shape and values.

        - Accepts either a single vector or a batch of vectors
        - Checks dimensionality equals settings.EMBEDDING_DIM
        - Ensures all values are finite
        """
        # Normalize to batch
        if not embeddings:
            return False
        batch: list[list[float]]
        if isinstance(embeddings[0], float):  # type: ignore[index]
            batch = [embeddings]  # type: ignore[list-item]
        else:
            batch = embeddings  # type: ignore[assignment]

        expected_dim = getattr(settings, "EMBEDDING_DIM", None)
        if not expected_dim:
            return False

        for vec in batch:
            if not isinstance(vec, list) or len(vec) != expected_dim:
                return False
            for v in vec:
                if not isfinite(v):
                    return False
        return True

    @staticmethod
    def check_vector_store_integrity(db: Session) -> IntegrityReport:
        """Run basic integrity checks on the vector store.

        Checks:
        - Total rows in content_embeddings
        - Null vectors count
        - Vector dimension mismatches via vector_dims (if available)
        - Presence of HNSW index on content_embeddings.content_vector
        """
        notes: list[str] = []
        total_rows = null_vectors = wrong_dims = 0
        hnsw_index_present: bool | None = None

        try:
            total_rows = int(
                db.execute(text("SELECT COUNT(*) FROM content_embeddings")).scalar_one() or 0
            )
            null_vectors = int(
                db.execute(
                    text("SELECT COUNT(*) FROM content_embeddings WHERE content_vector IS NULL")
                ).scalar_one()
                or 0
            )
        except Exception as exc:
            notes.append(f"count_failed:{type(exc).__name__}")

        # vector_dims check (optional)
        try:
            dim = int(getattr(settings, "EMBEDDING_DIM", 0))
            wrong_dims = int(
                db.execute(
                    text(
                        "SELECT COUNT(*) FROM content_embeddings "
                        "WHERE content_vector IS NOT NULL AND vector_dims(content_vector) != :dim"
                    ),
                    {"dim": dim},
                ).scalar_one()
                or 0
            )
        except Exception:
            wrong_dims = None
            notes.append("vector_dims_unavailable")

        # HNSW index presence (optional)
        try:
            row = db.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public' AND tablename = 'content_embeddings' "
                    "AND indexdef ILIKE '%USING hnsw%'"
                )
            ).first()
            hnsw_index_present = bool(row)
        except Exception:
            hnsw_index_present = None
            notes.append("hnsw_check_failed")

        ok = True
        if total_rows > 0 and null_vectors > 0:
            ok = False
            notes.append("has_null_vectors")
        if wrong_dims not in (None, 0):
            ok = False
            notes.append("has_wrong_dims")

        return IntegrityReport(
            ok=ok,
            total_rows=total_rows,
            null_vectors=null_vectors,
            wrong_dims=wrong_dims,
            hnsw_index_present=hnsw_index_present,
            notes=notes,
        )
