"""
Core ingestion engine and interfaces.

This module defines the minimal skeleton for a pluggable ingestion pipeline:
- `FormatProcessor`: interface for parsing raw sources (e.g., PDF, DOCX) into text
- `AdaptiveChunker`: interface for splitting text into semantically meaningful chunks
- `IngestionEngine`: orchestrates processing and chunking with progress callbacks and
  defensive error handling.

No external deps are introduced. Integrate real processors/chunkers in subsequent work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from app.core.logging_config import get_logger

# ----------------------------
# Exceptions
# ----------------------------


class IngestionError(Exception):
    """Generic ingestion failure wrapper."""


class UnsupportedFormatError(IngestionError):
    """Raised when no available `FormatProcessor` supports the given input."""


class ProcessingError(IngestionError):
    """Raised when a processor fails to parse the input into text."""


# ----------------------------
# Interfaces
# ----------------------------


class FormatProcessor(Protocol):
    """
    Interface for converting an input source (file path, URL, etc.) into raw text.
    """

    def supports(self, file_path: Path) -> bool:
        """Return True if this processor can handle the provided file path."""

    def process(self, file_path: Path) -> str:
        """Parse the input and return raw text. Should raise on unrecoverable errors."""


class AdaptiveChunker(Protocol):
    """Interface for splitting raw text into semantically meaningful chunks."""

    def chunk(self, text: str, *, max_tokens: int = 512, overlap: int = 50) -> list[str]:
        """Return a list of chunk strings."""


class NLTKAdaptiveChunker:
    """Sentence-aware chunker with configurable overlap.

    Uses NLTK's punkt sentence tokenizer to preserve sentence boundaries, then
    packs sentences into chunks roughly bounded by max_tokens (interpreted as
    number of words for simplicity). Overlap is applied in words between
    adjacent chunks to maintain context.
    """

    def __init__(self, language: str = "english") -> None:
        try:
            import nltk

            # Prefer nltk tokenizer only if resources are present; otherwise fallback
            has_punkt = False
            has_punkt_tab = False
            try:
                nltk.data.find("tokenizers/punkt")
                has_punkt = True
            except LookupError:
                has_punkt = False
            try:
                # Some newer nltk releases require punkt_tab as well
                nltk.data.find(f"tokenizers/punkt_tab/{language}/")
                has_punkt_tab = True
            except LookupError:
                has_punkt_tab = False

            if has_punkt and has_punkt_tab:
                self._sent_tokenize = nltk.sent_tokenize
            else:
                # Minimal fallback: naive sentence split on periods
                self._sent_tokenize = lambda t: [s.strip() for s in t.split(".") if s.strip()]
        except Exception:  # pragma: no cover - fallback
            # Minimal fallback: naive sentence split on periods
            self._sent_tokenize = lambda t: [s.strip() for s in t.split(".") if s.strip()]

    def chunk(self, text: str, *, max_tokens: int = 512, overlap: int = 50) -> list[str]:
        # Tokenize into sentences with safe fallback
        try:
            sentences = self._sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        max_words = max(1, max_tokens)
        overlap_words = max(0, overlap)

        for sentence in sentences:
            words = sentence.split()
            if current_len + len(words) <= max_words:
                current.append(sentence)
                current_len += len(words)
            else:
                if current:
                    chunks.append(" ".join(current))
                    # prepare next buffer with overlap
                    if overlap_words > 0:
                        tail_words = " ".join(" ".join(current).split()[-overlap_words:])
                        current = [tail_words] if tail_words else []
                        current_len = len(tail_words.split())
                    else:
                        current = []
                        current_len = 0
                # start new chunk with this sentence
                current.append(sentence)
                current_len += len(words)

        if current:
            chunks.append(" ".join(current))

        return chunks


# ----------------------------
# Progress callbacks
# ----------------------------


ProgressStage = Literal["started", "processed", "chunked", "completed", "failed"]


@dataclass(frozen=True)
class IngestionProgress:
    stage: ProgressStage
    file_path: str
    details: dict[str, Any] | None = None


ProgressCallback = Callable[[IngestionProgress], None]


def _notify_progress(
    callbacks: Iterable[ProgressCallback] | None, event: IngestionProgress
) -> None:
    if not callbacks:
        return
    for callback in callbacks:
        try:
            callback(event)
        except Exception:
            # Callbacks must never break the main flow
            pass


# ----------------------------
# Ingestion engine
# ----------------------------


class IngestionEngine:
    """
    Orchestrates document ingestion using registered processors and a chunker.

    Responsibilities:
    - Validate input
    - Select an appropriate `FormatProcessor`
    - Convert to text and chunk it using `AdaptiveChunker`
    - Emit progress events
    - Provide defensive error handling with explicit exception types
    """

    def __init__(
        self,
        *,
        processors: list[FormatProcessor],
        chunker: AdaptiveChunker | None = None,
        progress_callbacks: list[ProgressCallback] | None = None,
    ) -> None:
        if not processors:
            raise ValueError("At least one FormatProcessor must be provided")
        self._processors = processors
        self._chunker = chunker or NLTKAdaptiveChunker()
        self._progress_callbacks = progress_callbacks or []
        self._logger = get_logger(__name__)

    def ingest_file(
        self, file_path: str | Path, *, callbacks: list[ProgressCallback] | None = None
    ) -> list[str]:
        """
        Ingest a file and return chunked text.

        Raises:
            UnsupportedFormatError: If no processor can handle the file
            ProcessingError: If parsing fails
            IngestionError: For any other ingestion-related failures
        """
        file_path = Path(file_path)

        # Emit started
        _notify_progress(
            callbacks or self._progress_callbacks,
            IngestionProgress(stage="started", file_path=str(file_path)),
        )

        try:
            if not file_path.exists() or not file_path.is_file():
                msg = f"Input file does not exist: {file_path}"
                self._logger.error(msg)
                raise IngestionError(msg)

            processor = next((p for p in self._processors if p.supports(file_path)), None)
            if processor is None:
                msg = f"No processor available for file: {file_path}"
                self._logger.warning(msg)
                raise UnsupportedFormatError(msg)

            try:
                text = processor.process(file_path)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.exception("Processor failed", file=str(file_path))
                raise ProcessingError(str(exc)) from exc

            _notify_progress(
                callbacks or self._progress_callbacks,
                IngestionProgress(
                    stage="processed", file_path=str(file_path), details={"length": len(text)}
                ),
            )

            chunks = self._chunker.chunk(text)
            _notify_progress(
                callbacks or self._progress_callbacks,
                IngestionProgress(
                    stage="chunked",
                    file_path=str(file_path),
                    details={"num_chunks": len(chunks)},
                ),
            )

            _notify_progress(
                callbacks or self._progress_callbacks,
                IngestionProgress(stage="completed", file_path=str(file_path)),
            )
            return chunks

        except IngestionError:
            _notify_progress(
                callbacks or self._progress_callbacks,
                IngestionProgress(stage="failed", file_path=str(file_path)),
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive
            _notify_progress(
                callbacks or self._progress_callbacks,
                IngestionProgress(stage="failed", file_path=str(file_path)),
            )
            self._logger.exception("Unexpected ingestion failure")
            raise IngestionError(str(exc)) from exc
