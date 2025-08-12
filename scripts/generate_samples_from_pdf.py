#!/usr/bin/env python3
"""
Generate sample policy documents from a source PDF.

Usage:
  python scripts/generate_samples_from_pdf.py \
      "/Users/emre/GitHub/chatbot-api-service/pdf_documents/Hürriyet Partisi Tüzüğü v3.pdf"

Outputs written under: pdf_documents/samples/
 - First page PDF
 - First 3 pages PDF (if available)
 - Excerpt 5 pages starting at page 10 (if available)
 - Full text extraction
 - Excerpt text (first 3 pages)
"""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_pdf_pages(reader: PdfReader, page_indexes: list[int], out_path: Path) -> None:
    writer = PdfWriter()
    max_idx = len(reader.pages) - 1
    for idx in page_indexes:
        if 0 <= idx <= max_idx:
            writer.add_page(reader.pages[idx])
    if len(writer.pages) == 0:
        return
    with out_path.open("wb") as f:
        writer.write(f)


def extract_text_pages(reader: PdfReader, page_indexes: list[int]) -> str:
    texts: list[str] = []
    max_idx = len(reader.pages) - 1
    for idx in page_indexes:
        if 0 <= idx <= max_idx:
            try:
                texts.append(reader.pages[idx].extract_text() or "")
            except Exception:
                # Some PDFs may have extraction issues; continue best-effort
                texts.append("")
    return "\n\n".join(texts).strip()


def extract_text_all(reader: PdfReader) -> str:
    texts: list[str] = []
    for i in range(len(reader.pages)):
        try:
            texts.append(reader.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts).strip()


def write_generation_notice(out_dir: Path, stem: str) -> None:
    notice = (
        "This file is generated for tests and may contain Turkish words that codespell flags.\n"
        "It is intentionally excluded via pre-commit configuration.\n"
    )
    (out_dir / f"{stem}_README.txt").write_text(notice, encoding="utf-8")


def write_cover_page(out_dir: Path, stem: str, source_pdf: Path, total_pages: int) -> None:
    cover_path = out_dir / f"{stem}_cover.pdf"
    c = canvas.Canvas(str(cover_path), pagesize=A4)
    c.setFont("Helvetica", 14)
    text = c.beginText(2 * cm, 27 * cm)
    text.textLine("Generated Samples")
    text.textLine("")
    text.textLine(f"Source: {source_pdf.name}")
    text.textLine(f"Pages: {total_pages}")
    text.textLine("Note: Generated for tests; excluded from codespell.")
    c.drawText(text)
    c.showPage()
    c.save()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_samples_from_pdf.py <source_pdf_path>")
        sys.exit(1)

    source_pdf = Path(sys.argv[1])
    if not source_pdf.exists():
        print(f"Source PDF not found: {source_pdf}")
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "pdf_documents" / "samples"
    ensure_dir(out_dir)

    reader = PdfReader(str(source_pdf))
    total_pages = len(reader.pages)

    # Base stem for output file names
    stem = source_pdf.stem.lower().replace(" ", "_")

    # 1) First page PDF
    write_pdf_pages(reader, [0], out_dir / f"{stem}_page1.pdf")

    # 2) First 3 pages PDF (if available)
    write_pdf_pages(reader, [0, 1, 2], out_dir / f"{stem}_pages1-3.pdf")

    # 3) Excerpt 5 pages starting at page 10 (0-based index 9..13)
    excerpt_start = 9
    excerpt_pages = list(range(excerpt_start, excerpt_start + 5))
    write_pdf_pages(reader, excerpt_pages, out_dir / f"{stem}_excerpt_p10-14.pdf")

    # 4) Full text extraction
    full_text = extract_text_all(reader)
    (out_dir / f"{stem}_full.txt").write_text(full_text, encoding="utf-8")

    # 5) Excerpt text (first 3 pages)
    excerpt_text = extract_text_pages(reader, [0, 1, 2])
    (out_dir / f"{stem}_excerpt_pages1-3.txt").write_text(excerpt_text, encoding="utf-8")

    # 6) Add a small README and a cover page PDF to clarify generation context
    write_generation_notice(out_dir, stem)
    write_cover_page(out_dir, stem, source_pdf, total_pages)

    print("=== Sample generation complete ===")
    print(f"Source: {source_pdf}")
    print(f"Pages: {total_pages}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
