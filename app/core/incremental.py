from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dedup import ChunkInput, ContentDeduplicator
from app.core.embedding_manager import get_embedding_manager
from app.core.embedding_storage import get_embedding_storage
from app.core.ingestion import NLTKAdaptiveChunker
from app.core.logging_config import get_logger
from app.core.metadata import ChunkMetadataExtractor, summarize_headers
from app.core.metrics import get_metrics_backend
from app.db.models import ContentEmbedding, Document, DocumentChunk


@dataclass(frozen=True)
class ChangeSet:
    """Represents detected changes for a document update.

    - If no `changed_pages` are provided, the entire document is considered changed.
    """

    is_new_version: bool
    changed_pages: list[int]
    reason: str | None = None


class IncrementalProcessor:
    """Handles incremental updates and selective re-embedding.

    This initial implementation provides:
    - Content-hash based change detection
    - Optional page-level targeting when caller supplies page hashes
    - Selective re-embedding for changed pages using LlamaIndex PGVectorStore

    Notes:
    - We rely on `content_embeddings` table as the vector store sink.
    - Page boundaries are respected via the `page_number` field.
    - Chunking within a page uses the same sentence-aware chunker as ingestion.
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._chunker = NLTKAdaptiveChunker()
        self._metrics = get_metrics_backend()
        self._meta_extractor = ChunkMetadataExtractor()

    def detect_changes(
        self,
        *,
        db: Session,
        document_id: str,
        new_content_hash: str,
        new_page_hashes: Mapping[int, str] | None = None,
    ) -> ChangeSet:
        """Detect what changed for a document update.

        Strategy:
        - If the document content hash is identical, return no changes
        - If `new_page_hashes` are provided, treat those pages as changed (best-effort v1)
        - Else, fall back to full-document change
        """

        document = db.get(Document, document_id)
        if document is None:
            return ChangeSet(is_new_version=True, changed_pages=[], reason="document_not_found")

        if document.content_hash == new_content_hash:
            return ChangeSet(is_new_version=False, changed_pages=[], reason="no_change")

        # v1 heuristic: if caller provides page hashes, assume those pages changed
        # (future: compare against stored per-page digests when available)
        if new_page_hashes:
            changed_pages = sorted({p for p, h in new_page_hashes.items() if h})
        else:
            # Fallback: treat all pages as changed
            page_count = document.page_count or 0
            changed_pages = list(range(1, page_count + 1)) if page_count > 0 else []

        return ChangeSet(is_new_version=True, changed_pages=changed_pages, reason="hash_changed")

    def process_incremental_update(
        self,
        *,
        db: Session,
        document_id: str,
        changes: ChangeSet,
        new_content_hash: str | None = None,
        new_page_texts: Mapping[int, str] | None = None,
    ) -> None:
        """Process only the changed portions of a document.

        - Deletes existing vectors for the changed pages
        - Re-chunks and re-embeds provided new page texts
        - Bumps document version and updates `content_hash`

        Caller must provide `new_page_texts` for the pages marked as changed.
        """

        document = db.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        if not changes.changed_pages:
            # Nothing to do
            return

        if new_page_texts is None:
            raise ValueError("new_page_texts is required for selective re-embedding")

        # 1) Document-level dedup/versioning
        source_document = document.filename or document.original_filename or document.id
        if new_content_hash:
            try:
                # Use existing document attributes to drive dedup/version bump
                dedup = ContentDeduplicator()
                _doc, _action = dedup.upsert_document_by_hash(
                    db,
                    filename=document.filename,
                    storage_uri=document.storage_uri,
                    mime_type=document.mime_type,
                    file_size=document.file_size,
                    page_count=document.page_count,
                    new_content_hash=new_content_hash,
                    language=document.language or "en",
                    created_by=document.created_by,
                    tags=getattr(document, "tags", None),
                )
            except Exception:
                # Defensive: proceed even if dedup path fails
                pass

        # 2) Delete existing vectors for changed pages
        (
            db.query(ContentEmbedding)
            .filter(
                ContentEmbedding.source_document == source_document,
                ContentEmbedding.page_number.in_(changes.changed_pages),
            )
            .delete(synchronize_session=False)
        )

        # 3) Re-embed only changed pages
        from time import time

        t0 = time()
        page_text_map = {p: new_page_texts.get(p, "") for p in changes.changed_pages}

        self._logger.info(
            "Starting incremental re-embedding",
            document_id=document_id,
            changed_pages=changes.changed_pages,
            page_count=len(page_text_map),
            total_text_length=sum(len(text) for text in page_text_map.values()),
        )

        self._reembed_pages(page_texts=page_text_map, source_document=source_document)
        embedding_duration = time() - t0

        self._metrics.record_histogram(
            "embedding_latency_seconds", embedding_duration, {"stage": "incremental_reembed"}
        )

        self._logger.info(
            "Incremental re-embedding completed",
            document_id=document_id,
            changed_pages=changes.changed_pages,
            embedding_duration_seconds=round(embedding_duration, 3),
        )

        # Also record page counters when we have texts
        pages_count = sum(1 for _, txt in page_text_map.items() if (txt or "").strip())
        if pages_count:
            from app.core.metrics import IngestionMetrics

            ingest_metrics = IngestionMetrics(self._metrics.backend)
            ingest_metrics.increment_pages_processed(status="reembed", count=pages_count)
        # 4) Upsert chunks for changed pages (acceptance: inserts for new, updates for changed)
        self._upsert_changed_page_chunks(
            db=db,
            document_id=document_id,
            changed_pages=changes.changed_pages,
            new_page_texts=new_page_texts,
        )
        # Dedup already handled version/hash; no direct doc mutation here

    def _upsert_changed_page_chunks(
        self,
        *,
        db: Session,
        document_id: str,
        changed_pages: list[int],
        new_page_texts: Mapping[int, str],
    ) -> None:
        """Upsert chunks for changed pages with structured logging."""
        try:
            dedup = ContentDeduplicator()
            for page in changed_pages:
                text = new_page_texts.get(page, "")
                if not text:
                    continue
                prepared = self._prepare_chunk_inputs(
                    db=db, document_id=document_id, page_number=page, text=text
                )
                if prepared:
                    res = dedup.upsert_chunks(db, document_id=document_id, chunks=prepared)
                    self._logger.info(
                        "chunks_upserted",
                        extra={
                            "document_id": document_id,
                            "page_number": page,
                            "inserted": res.get("inserted", 0),
                            "updated": res.get("updated", 0),
                            "unchanged": res.get("unchanged", 0),
                        },
                    )
        except Exception:
            # Defensive: chunk upsert failures shouldn't break re-embedding path
            pass

    def _prepare_chunk_inputs(
        self, *, db: Session, document_id: str, page_number: int, text: str
    ) -> list[ChunkInput]:
        """Chunk page text and map to stable indices based on existing rows.

        Strategy:
        - Enumerate new chunks for the page
        - Fetch existing chunks for (document_id, page_number) ordered by index
        - Assign existing indices to the first N new chunks
        - If more new chunks than existing, append with next indices (inserts)
        - If fewer new chunks than existing, do not delete extras
        """
        new_chunks = self._chunker.chunk(text)
        if not new_chunks:
            return []
        # Derive page-level metadata once
        page_meta = self._meta_extractor.extract_page_metadata(
            page_number=page_number, page_text=text
        )
        if page_meta.headers:
            self._logger.info(
                "page_metadata_detected",
                extra={
                    "document_id": document_id,
                    "page_number": page_number,
                    "headers": summarize_headers(page_meta.headers),
                },
            )
        existing = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.page_number == page_number,
            )
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        prepared: list[ChunkInput] = []
        for i, content in enumerate(new_chunks):
            if i < len(existing):
                idx = existing[i].chunk_index
            else:
                last_idx = existing[-1].chunk_index if existing else -1
                idx = last_idx + 1 + (i - len(existing))
            prepared.append(
                ChunkInput(
                    index=idx,
                    content=content,
                    page_number=page_number,
                    section_title=self._meta_extractor.assign_section_title(page_meta, content),
                    extra_metadata=self._meta_extractor.build_chunk_extra_metadata(page_meta),
                )
            )
        return prepared

    def _reembed_pages(self, *, page_texts: Mapping[int, str], source_document: str) -> None:
        """Chunk and embed provided page texts, storing vectors via dimension-aware storage.

        Uses embedding manager with provider fallback and dimension-aware table routing.
        """

        from llama_index.core.node_parser import SentenceSplitter

        embedding_manager = get_embedding_manager()
        self._log_provider_status(embedding_manager, source_document)

        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        llama_docs = self._build_llama_documents(
            page_texts=page_texts, source_document=source_document
        )

        if not llama_docs:
            self._log_no_docs(source_document=source_document, page_texts=page_texts)
            return

        self._log_chunking_start(source_document=source_document, doc_count=len(llama_docs))
        nodes = node_parser.get_nodes_from_documents(llama_docs)

        texts, node_refs = self._extract_texts_and_refs(nodes, source_document)
        if not texts:
            self._log_no_chunks(source_document=source_document, node_count=len(nodes))
            return

        embeddings = self._embed_texts(embedding_manager, texts, source_document)
        for n, e in zip(node_refs, embeddings, strict=True):
            n.embedding = e

        active_provider = self._select_active_provider(embedding_manager)
        self._store_embeddings(node_refs, source_document, active_provider)

    def _log_provider_status(self, embedding_manager: Any, source_document: str) -> None:
        provider_status = embedding_manager.get_provider_status()
        self._logger.info(
            "embedding_provider_selected",
            extra={
                "provider_status": provider_status,
                "source_document": source_document,
            },
        )

    def _build_llama_documents(
        self, *, page_texts: Mapping[int, str], source_document: str
    ) -> list[Any]:
        from llama_index.core import Document as LlamaDocument

        llama_docs: list[LlamaDocument] = []
        for page, text in page_texts.items():
            sanitized = _sanitize_text(text)
            if not sanitized:
                self._logger.debug(
                    "Skipping empty page text", source_document=source_document, page_number=page
                )
                continue
            llama_docs.append(
                LlamaDocument(
                    text=sanitized,
                    metadata={
                        "source_document": source_document,
                        "page_number": page,
                    },
                )
            )
        return llama_docs

    def _log_no_docs(self, *, source_document: str, page_texts: Mapping[int, str]) -> None:
        self._logger.warning(
            "No valid documents to embed",
            source_document=source_document,
            page_count=len(page_texts),
        )

    def _log_chunking_start(self, *, source_document: str, doc_count: int) -> None:
        self._logger.info(
            "Starting document chunking",
            source_document=source_document,
            document_count=doc_count,
            chunk_size=512,
            chunk_overlap=20,
        )

    def _extract_texts_and_refs(
        self, nodes: list[Any], source_document: str
    ) -> tuple[list[str], list[Any]]:
        texts: list[str] = []
        node_refs: list[Any] = []
        for n in nodes:
            content = (n.get_content() or "").strip()
            if content:
                texts.append(content)
                node_refs.append(n)
        return texts, node_refs

    def _log_no_chunks(self, *, source_document: str, node_count: int) -> None:
        self._logger.warning(
            "No valid chunks after parsing",
            source_document=source_document,
            node_count=node_count,
        )

    def _embed_texts(
        self, embedding_manager: Any, texts: list[str], source_document: str
    ) -> list[Any]:
        from time import time

        self._logger.info(
            "Starting batch embedding generation",
            source_document=source_document,
            chunk_count=len(texts),
            total_text_length=sum(len(text) for text in texts),
            avg_chunk_length=round(sum(len(text) for text in texts) / len(texts), 1),
        )

        embed_start = time()
        embeddings = embedding_manager.get_text_embeddings(texts)
        embed_duration = time() - embed_start

        self._logger.info(
            "Batch embedding generation completed",
            source_document=source_document,
            chunk_count=len(texts),
            embedding_time_seconds=round(embed_duration, 3),
            embeddings_per_second=round(len(texts) / embed_duration, 2)
            if embed_duration > 0
            else 0,
        )
        return embeddings

    def _select_active_provider(self, embedding_manager: Any) -> str:
        provider_status = embedding_manager.get_provider_status()
        for provider, status in provider_status.items():
            if status.get("available", False):
                return provider
        return "unknown"

    def _store_embeddings(
        self, node_refs: list[Any], source_document: str, active_provider: str
    ) -> None:
        from time import time

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        store_start = time()

        embeddings_data = []
        for node in node_refs:
            page_number = node.metadata.get("page_number", 1)
            content_text = node.text
            embedding_vector = node.embedding
            embeddings_data.append((source_document, page_number, content_text, embedding_vector))

        self._logger.info(
            "Starting dimension-aware vector store insertion",
            source_document=source_document,
            node_count=len(node_refs),
            embedding_dimension=len(embeddings_data[0][3]) if embeddings_data else 0,
            active_provider=active_provider,
        )

        engine = create_engine(settings.DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as storage_session:
            embedding_storage = get_embedding_storage(storage_session)
            stored_count = embedding_storage.store_embeddings(embeddings_data, active_provider)

        store_duration = time() - store_start
        self._logger.info(
            "Dimension-aware vector store insertion completed",
            source_document=source_document,
            stored_count=stored_count,
            insertion_time_seconds=round(store_duration, 3),
            embedding_dimension=len(embeddings_data[0][3]) if embeddings_data else 0,
            active_provider=active_provider,
        )


def _sanitize_text(text: str) -> str:
    """Normalize whitespace and strip empties for stable embedding input."""
    if not text:
        return ""
    sanitized = " ".join(text.split())
    return sanitized.strip()
