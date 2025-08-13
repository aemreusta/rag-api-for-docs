from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def _is_likely_header(line: str) -> bool:
    """Heuristic to detect header-like lines.

    Rules (v1, conservative):
    - Non-empty, short lines (<= 120 chars)
    - Either all-caps alpha (with digits and spaces allowed), or
      starts with an outline prefix like '1.', '1.1', 'I.', 'A)'
    - Avoid lines that end with sentence punctuation typical for prose
    """
    candidate = line.strip()
    if not candidate:
        return False
    if len(candidate) > 120:
        return False

    terminal = candidate[-1]
    if terminal in {".", "?", "!"} and len(candidate.split()) > 6:
        # Likely a sentence
        return False

    # Outline style prefixes
    import re

    outline_prefix = re.match(
        r"^(?:\d+(?:[.\)])(?:\d+[.\)])*\s+|[IVXLCM]+\.|[A-Z]\)\s+)", candidate
    )
    if outline_prefix:
        return True

    # All caps (allow digits, spaces, hyphens, ampersands)
    letters = [c for c in candidate if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True

    return False


def _extract_header_lines(page_text: str) -> list[str]:
    """Return unique header-like lines in their original order for a page."""
    seen: set[str] = set()
    headers: list[str] = []
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_likely_header(line):
            key = line.lower()
            if key not in seen:
                seen.add(key)
                headers.append(line)
    return headers


@dataclass(frozen=True)
class PageMetadata:
    page_number: int
    headers: list[str]
    section_title: str | None


class ChunkMetadataExtractor:
    """Derives per-page metadata and provides helpers for chunk annotation.

    Strategy:
    - Detect header-like lines on the page using simple heuristics
    - The page's `section_title` is the first detected header if any
    - All chunks on the page inherit the same `section_title`
    - Extra metadata carries the discovered `headers` list
    """

    def extract_page_metadata(self, *, page_number: int, page_text: str) -> PageMetadata:
        headers = _extract_header_lines(page_text)
        section_title = headers[0] if headers else None
        return PageMetadata(page_number=page_number, headers=headers, section_title=section_title)

    def build_chunk_extra_metadata(self, page_meta: PageMetadata) -> dict:
        return {
            "page_number": page_meta.page_number,
            "headers": page_meta.headers,
        }

    def assign_section_title(self, page_meta: PageMetadata, _chunk_text: str) -> str | None:
        # v1: same title for all chunks on the page
        return page_meta.section_title


def summarize_headers(headers: Iterable[str]) -> str:
    """Return a short summary string for logging/metrics."""
    headers = list(headers)
    if not headers:
        return "none"
    show = headers[:3]
    suffix = "" if len(headers) <= 3 else f" +{len(headers) - 3} more"
    return " | ".join(show) + suffix
