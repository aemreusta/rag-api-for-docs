# Ingestion Engine (skeleton)

Purpose:

- Provide a pluggable ingestion pipeline with clean interfaces for parsing and chunking
- Emit progress callbacks for observability without coupling to specific backends

Key pieces:

- `FormatProcessor`: converts file input into raw text (`supports()`, `process()`)
- `AdaptiveChunker`: splits text into chunks (`chunk()`)
- `IngestionEngine`: orchestrates, validates, and emits progress events (`started`, `processed`, `chunked`, `completed`, `failed`)

Next steps:

- Implement concrete processors for PDF/DOCX/URL
- Replace the simple test chunker with an adaptive semantic chunker
- Connect metrics/logging hooks where needed and update Mermaid diagrams if architecture changes
