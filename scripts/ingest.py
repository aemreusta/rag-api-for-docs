import logging

import psycopg2  # Used to check for the extension
from llama_index.callbacks.langfuse.base import LlamaIndexCallbackHandler
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

    # Create a trace for the ingestion process
    langfuse_handler.start_trace(name="pdf-ingestion", metadata={"source_directory": PDF_DIRECTORY})
    # --- End Langfuse Setup ---

    logger.info("Starting data ingestion process...")

    # Verify pgvector extension is enabled
    logger.info("Checking pgvector extension...")
    conn = psycopg2.connect(settings.DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if cur.fetchone() is None:
        logger.error("pgvector extension not found. Please enable it in your database.")
        langfuse_handler.end_trace(
            output={"status": "error", "error": "pgvector extension not found"}
        )
        langfuse_handler.flush()
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

    # Set up the PGVectorStore
    logger.info("Setting up PostgreSQL vector store...")
    vector_store = PGVectorStore.from_params(
        dsn=settings.DATABASE_URL,
        table_name="charity_policies",
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

    # At the end of the script
    langfuse_handler.end_trace(output={"status": "success", "documents_indexed": len(documents)})
    langfuse_handler.flush()  # Ensure all data is sent
    logger.info("Ingestion complete. Trace sent to Langfuse.")


if __name__ == "__main__":
    main()
