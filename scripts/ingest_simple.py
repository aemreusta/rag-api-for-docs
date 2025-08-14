import logging
import os
import time

import psycopg2  # Used to check for the extension
from llama_index.core import (
    Settings as LlamaSettings,
)
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings


class RetryingEmbedding:
    """Thin wrapper to add retry/backoff around an embedding model.

    Handles Google 429 ResourceExhausted by exponential backoff.
    """

    def __init__(self, inner, max_retries: int = 6, initial_delay: float = 1.0):
        self.inner = inner
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def get_text_embedding(self, text: str):
        delay = self.initial_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.inner.get_text_embedding(text)
            except Exception as e:  # pragma: no cover
                # Detect Google quota error conservatively by message
                msg = str(e)
                if "Resource has been exhausted" in msg or "429" in msg:
                    logger.warning(
                        f"Embedding rate/quota hit (attempt {attempt}); sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
        # Final attempt
        return self.inner.get_text_embedding(text)

    def get_text_embedding_batch(self, texts: list[str]):
        return [self.get_text_embedding(t) for t in texts]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_DIRECTORY = "pdf_documents/"


def _normalize_dsn(url: str) -> str:
    """Accept SQLAlchemy-style URLs (postgresql+psycopg2://...) and return psycopg2 DSN."""
    if "+" in url and url.startswith("postgresql+"):
        # Convert to plain postgresql://
        url = url.replace("postgresql+psycopg2://", "postgresql://")
    return url


def _sanitize_text(text: str) -> str:
    """Remove non-printable characters and trim whitespace."""
    if text is None:
        return ""
    sanitized = "".join(ch for ch in text if ch.isprintable())
    return sanitized.strip()


def main():
    if not settings.INGEST_PARALLEL_DEPLOYMENT:
        logger.warning(
            "Legacy ingest path is deprecated. Use /api/v1/docs endpoints. "
            "Set INGEST_PARALLEL_DEPLOYMENT=true to run in dual mode during migration."
        )
    logger.info("Starting simplified data ingestion process...")

    # Verify pgvector extension is enabled
    logger.info("Checking pgvector extension...")
    conn = psycopg2.connect(_normalize_dsn(settings.DATABASE_URL))
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if cur.fetchone() is None:
        logger.error("pgvector extension not found. Please enable it in your database.")
        conn.close()
        return
    conn.close()
    logger.info("pgvector extension verified")

    # Set up LlamaIndex LLM and Embedding Model
    logger.info("Setting up LLM and embedding models...")
    LlamaSettings.llm = OpenRouter(
        api_key=settings.OPENROUTER_API_KEY, model=settings.LLM_MODEL_NAME
    )
    # Use Gemini embeddings by default for ingestion
    from app.core.embeddings import get_embedding_model

    LlamaSettings.embed_model = get_embedding_model()

    # Load documents from the PDF directory
    logger.info(f"Loading documents from {PDF_DIRECTORY}...")
    reader = SimpleDirectoryReader(
        input_dir=PDF_DIRECTORY,
        required_exts=[".pdf"],
        exclude_hidden=True,
        recursive=True,
    )
    documents = reader.load_data()
    max_docs = int(os.getenv("INGEST_MAX_DOCS", "60"))
    if len(documents) > max_docs:
        documents = documents[:max_docs]
        logger.warning(f"Too many documents for current quota; trimming to first {max_docs}.")
    logger.info(f"Loaded {len(documents)} document(s) from '{PDF_DIRECTORY}'.")

    # Set up the PGVectorStore with correct parameters
    logger.info("Setting up PostgreSQL vector store...")
    vector_store = PGVectorStore.from_params(
        database=settings.POSTGRES_DB,
        host=settings.POSTGRES_SERVER,
        password=settings.POSTGRES_PASSWORD,
        port=5432,
        user=settings.POSTGRES_USER,
        table_name="content_embeddings",
        embed_dim=settings.EMBEDDING_DIM,  # Must match DB (1536 for Gemini default)
    )

    # Build the index (manual embedding to avoid remote provider)
    logger.info("Creating vector index and storing embeddings...")
    StorageContext.from_defaults(vector_store=vector_store)
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # Convert documents to nodes
    nodes = node_parser.get_nodes_from_documents(documents)
    # Prepare (node, sanitized_text) pairs and drop empties to avoid provider errors
    node_text_pairs = []
    for n in nodes:
        text = _sanitize_text(n.get_content() or "")
        if text and len(text) >= 3:
            node_text_pairs.append((n, text))
    if not node_text_pairs:
        logger.warning("No non-empty chunks found after parsing. Nothing to ingest.")
        return
    # Compute embeddings using the configured ingestion embed model
    embedder = LlamaSettings.embed_model
    texts = [t for _, t in node_text_pairs]
    # Final safety: drop any residual empties
    bad_idx = [i for i, t in enumerate(texts) if not t or not t.strip()]
    if bad_idx:
        logger.warning(
            f"Dropping {len(bad_idx)} empty chunks before embedding: idx={bad_idx[:5]}..."
        )
        texts = [t for i, t in enumerate(texts) if i not in bad_idx]
        node_text_pairs = [p for i, p in enumerate(node_text_pairs) if i not in bad_idx]
        if not texts:
            logger.warning("No embeddable text remains after filtering. Exiting.")
            return
    embeddings = embedder.get_text_embedding_batch(texts)
    for (n, _), e in zip(node_text_pairs, embeddings, strict=True):
        n.embedding = e
    # Add to vector store directly
    vector_store.add(nodes)
    logger.info("Successfully built index and stored it in PostgreSQL.")
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
