from __future__ import annotations

from app.core.metadata import ChunkMetadataExtractor


def test_metadata_extractor_detects_headers_and_section_title():
    page_text = (
        "BÖLÜM I\n"
        "Genel Hükümler\n"
        "\n"
        "Bu bölüm, tüzüğün kapsamını açıklar. Cümleler devam eder.\n"
        "Madde 1. Amaç ve Kapsam.\n"
    )

    extractor = ChunkMetadataExtractor()
    meta = extractor.extract_page_metadata(page_number=1, page_text=page_text)

    assert meta.page_number == 1
    # First header-like line becomes the section title
    assert meta.section_title is not None
    assert meta.headers and meta.section_title == meta.headers[0]

    extra = extractor.build_chunk_extra_metadata(meta)
    assert extra.get("page_number") == 1
    assert isinstance(extra.get("headers"), list)

    # All chunks on the page share the same section title (v1 policy)
    assert extractor.assign_section_title(meta, "chunk text") == meta.section_title
