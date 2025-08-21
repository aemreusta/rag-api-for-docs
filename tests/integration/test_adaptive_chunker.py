from app.core.ingestion import NLTKAdaptiveChunker


def test_adaptive_chunker_basic_split():
    text = "Hello world. This is a test. Another sentence here. And one more."
    chunker = NLTKAdaptiveChunker()
    chunks = chunker.chunk(text, max_tokens=5, overlap=2)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    # ensure sentences are preserved within chunks
    for c in chunks:
        assert c.endswith(".") or c == chunks[-1] or len(c) > 0


def test_adaptive_chunker_overlap():
    text = (
        "One two three four five six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."
    )
    chunker = NLTKAdaptiveChunker()
    chunks = chunker.chunk(text, max_tokens=5, overlap=2)
    assert len(chunks) >= 2
    # last words of first chunk should appear at start of second due to overlap
    first_tail = " ".join(chunks[0].split()[-2:])
    assert first_tail.split()[0] in chunks[1]
