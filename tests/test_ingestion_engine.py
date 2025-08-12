from pathlib import Path

import pytest

from app.core.ingestion import (
    IngestionEngine,
    IngestionError,
    IngestionProgress,
    UnsupportedFormatError,
)


class TxtOnlyProcessor:
    def supports(self, file_path: Path) -> bool:  # type: ignore[override]
        return file_path.suffix.lower() == ".txt"

    def process(self, file_path: Path) -> str:  # type: ignore[override]
        return file_path.read_text(encoding="utf-8")


class EchoChunker:
    def chunk(self, text: str, *, max_tokens: int = 512, overlap: int = 50) -> list[str]:  # type: ignore[override]
        # naive split by whitespace for tests
        return [token for token in text.strip().split() if token]


def test_ingest_file_happy_path(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world from ingestion", encoding="utf-8")

    progress_events: list[IngestionProgress] = []

    def on_progress(evt: IngestionProgress) -> None:
        progress_events.append(evt)

    engine = IngestionEngine(
        processors=[TxtOnlyProcessor()],
        chunker=EchoChunker(),
        progress_callbacks=[on_progress],
    )

    chunks = engine.ingest_file(sample)

    assert chunks == ["hello", "world", "from", "ingestion"]
    # Verify meaningful progress transitions
    stages = [e.stage for e in progress_events]
    assert stages[0] == "started"
    assert "processed" in stages
    assert "chunked" in stages
    assert stages[-1] == "completed"


def test_unsupported_format_raises(tmp_path: Path):
    sample_pdf = tmp_path / "sample.pdf"
    sample_pdf.write_text("fake pdf bytes", encoding="utf-8")

    engine = IngestionEngine(processors=[TxtOnlyProcessor()], chunker=EchoChunker())

    with pytest.raises(UnsupportedFormatError):
        engine.ingest_file(sample_pdf)


def test_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "nope.txt"
    engine = IngestionEngine(processors=[TxtOnlyProcessor()], chunker=EchoChunker())

    with pytest.raises(IngestionError):
        engine.ingest_file(missing)


def test_progress_callback_errors_do_not_break(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world", encoding="utf-8")

    def bad_callback(evt: IngestionProgress) -> None:
        raise RuntimeError("boom")

    engine = IngestionEngine(
        processors=[TxtOnlyProcessor()],
        chunker=EchoChunker(),
        progress_callbacks=[bad_callback],
    )

    # Should not raise due to callback failure
    chunks = engine.ingest_file(sample)
    assert chunks == ["hello", "world"]
