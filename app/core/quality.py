from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None


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
