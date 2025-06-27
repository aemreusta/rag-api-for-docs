import logging
import sys
import types

import psycopg2  # Used to check for the extension

# Langfuse integration is optional at runtime and may be unavailable in some
# environments (e.g. CI).  Import it if present, otherwise fall back to a
# lightweight stub so that the rest of the module can still be imported and
# tested.

try:
    from llama_index.callbacks.langfuse.base import LlamaIndexCallbackHandler
except ModuleNotFoundError:  # pragma: no cover – optional dependency missing

    class LlamaIndexCallbackHandler:  # type: ignore
        """Stub replacement used when the real Langfuse integration is absent."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401,E501
            pass

    # Ensure that downstream `import` statements targeting the original package
    # path resolve to this stub to keep import-side effects in the test-suite
    # working even when the optional extra is not installed.
    stub_module = types.ModuleType("_langfuse_stub")
    stub_module.LlamaIndexCallbackHandler = LlamaIndexCallbackHandler
    sys.modules.setdefault("llama_index.callbacks.langfuse.base", stub_module)

from llama_index.core import (
    Settings as LlamaSettings,
)
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_DIRECTORY = "pdf_documents/"


def main():
    # --- Langfuse Setup ---
    langfuse_handler = LlamaIndexCallbackHandler(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
    LlamaSettings.callback_manager = CallbackManager([langfuse_handler])  # Add Langfuse handler

    logger.info("Starting data ingestion process...")

    # Verify pgvector extension is enabled
    logger.info("Checking pgvector extension...")
    conn = psycopg2.connect(settings.DATABASE_URL)
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
    # Using a default local embedding model for ingestion is fast and free
    LlamaSettings.embed_model = "local:BAAI/bge-small-en-v1.5"

    # Load documents from the PDF directory
    logger.info(f"Loading documents from {PDF_DIRECTORY}...")
    reader = SimpleDirectoryReader(input_dir=PDF_DIRECTORY)
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} document(s) from '{PDF_DIRECTORY}'.")

    # Set up the PGVectorStore with correct parameters
    logger.info("Setting up PostgreSQL vector store...")
    vector_store = PGVectorStore.from_params(
        database="app",
        host="postgres",
        password="postgres",
        port=5432,
        user="postgres",
        table_name="content_embeddings",
        embed_dim=384,  # Dimension for bge-small-en-v1.5
    )

    # Build the index
    logger.info("Creating vector index and storing embeddings...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        node_parser=node_parser,
        show_progress=True,
    )
    logger.info("Successfully built index and stored it in PostgreSQL.")
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
