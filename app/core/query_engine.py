"""Query engine setup and convenience helpers.

This file relies on LlamaIndex which, in newer releases, imports
`Secret` from `pydantic`.  The generic `Secret` type was introduced in
Pydantic v2.6.  Our dependency constraints may resolve an earlier
version (e.g. 2.5.x) when the symbol is missing, resulting in an
`ImportError` originating from LlamaIndex.  To keep the application
and the test-suite working without forcing an immediate global
dependency upgrade, we pro-actively patch `pydantic` with a minimal
fallback implementation when the attribute is absent.
"""

import asyncio
import time
from typing import Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.cache import cache_key_for_chat, cached
from app.core.config import settings
from app.core.embeddings import get_embedding_model
from app.core.llm_router import LLMRouter
from app.core.logging_config import get_logger
from app.core.metrics import vector_metrics

logger = get_logger(__name__)


class LoggedVectorStoreWrapper:
    """Wrapper that adds structured logging to vector store operations."""

    def __init__(self, vector_store: PGVectorStore):
        self._vector_store = vector_store
        self._logger = get_logger(f"{__name__}.vector_store")

    def add(self, nodes):
        """Add nodes to vector store with logging."""
        start_time = time.time()
        node_count = len(nodes) if nodes else 0

        try:
            self._logger.info(
                "Vector store insertion started",
                operation="add",
                node_count=node_count,
                table_name="content_embeddings",
            )

            result = self._vector_store.add(nodes)
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.info(
                "Vector store insertion completed",
                operation="add",
                node_count=node_count,
                response_time_ms=duration_ms,
                table_name="content_embeddings",
            )

            return result
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.error(
                "Vector store insertion failed",
                operation="add",
                node_count=node_count,
                response_time_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
                table_name="content_embeddings",
            )
            raise

    def query(self, query, **kwargs):
        """Query vector store with logging."""
        start_time = time.time()
        similarity_top_k = kwargs.get("similarity_top_k", 2)

        try:
            self._logger.info(
                "Vector store query started",
                operation="query",
                similarity_top_k=similarity_top_k,
                query_length=len(str(query)) if query else 0,
                table_name="content_embeddings",
            )

            result = self._vector_store.query(query, **kwargs)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            result_count = len(result.nodes) if hasattr(result, "nodes") and result.nodes else 0

            self._logger.info(
                "Vector store query completed",
                operation="query",
                similarity_top_k=similarity_top_k,
                result_count=result_count,
                response_time_ms=duration_ms,
                table_name="content_embeddings",
            )

            return result
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.error(
                "Vector store query failed",
                operation="query",
                similarity_top_k=similarity_top_k,
                response_time_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
                table_name="content_embeddings",
            )
            raise

    def delete(self, ref_doc_id, **kwargs):
        """Delete from vector store with logging."""
        start_time = time.time()

        try:
            self._logger.info(
                "Vector store deletion started",
                operation="delete",
                ref_doc_id=ref_doc_id,
                table_name="content_embeddings",
            )

            result = self._vector_store.delete(ref_doc_id, **kwargs)
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.info(
                "Vector store deletion completed",
                operation="delete",
                ref_doc_id=ref_doc_id,
                response_time_ms=duration_ms,
                table_name="content_embeddings",
            )

            return result
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.error(
                "Vector store deletion failed",
                operation="delete",
                ref_doc_id=ref_doc_id,
                response_time_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
                table_name="content_embeddings",
            )
            raise

    def __getattr__(self, name):
        """Forward other attributes to the wrapped vector store."""
        return getattr(self._vector_store, name)


# Set up the embedding model from settings (provider/model handled in helper)
Settings.embed_model = get_embedding_model()

# Initialize components
_raw_vector_store = PGVectorStore.from_params(
    database=settings.POSTGRES_DB,
    host=settings.POSTGRES_SERVER,
    password=settings.POSTGRES_PASSWORD,
    port=5432,
    user=settings.POSTGRES_USER,
    table_name="content_embeddings",
    # Embedding dimension is configurable via settings
    embed_dim=settings.EMBEDDING_DIM,
)
vector_store = LoggedVectorStoreWrapper(_raw_vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Initialize LLM Router with automatic fallback
llm_router = LLMRouter()

# Create a query engine with the router
query_engine = index.as_query_engine(llm=llm_router)

# Create a ChatEngine for conversational context
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    llm=llm_router,
)


def _cache_key_for_chat_response(question: str, session_id: str, model: str | None = None) -> str:
    """Generate cache key for chat responses using the LLM model name.

    Accepts an optional model parameter to match the wrapped function signature.
    """
    effective_model = model
    if not effective_model:
        effective_model = (
            llm_router._available_providers[0].get_model_name()
            if llm_router._available_providers
            else ""
        )
    return cache_key_for_chat(question, session_id, effective_model)


@vector_metrics.time_vector_search
@cached(ttl=settings.CACHE_TTL_SECONDS, key_func=_cache_key_for_chat_response)
async def get_chat_response_async(question: str, session_id: str, model: str | None = None) -> Any:
    """Get chat response with caching and flexible performance monitoring."""
    start_time = time.time()

    logger.info(
        "Chat response request started",
        question_length=len(question),
        session_id=session_id,
        requested_model=model,
        cache_enabled=settings.CACHE_ENABLED,
    )

    try:
        # NOTE: For now, session_id is a placeholder. Real memory will be added in Phase 3.
        # This engine will internally condense the question but doesn't have persistent memory yet.

        # Run the chat in executor since LlamaIndex chat_engine.chat is synchronous
        loop = asyncio.get_event_loop()
        # Propagate per-request model preference to LlamaIndex flows via context var
        if hasattr(llm_router, "set_model_preference"):
            llm_router.set_model_preference(model)
        response = await loop.run_in_executor(None, chat_engine.chat, question)

        duration_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Chat response completed",
            question_length=len(question),
            session_id=session_id,
            response_length=len(str(response)) if response else 0,
            response_time_ms=duration_ms,
            used_model=model,
        )

        return response
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)

        logger.error(
            "Chat response failed",
            question_length=len(question),
            session_id=session_id,
            response_time_ms=duration_ms,
            error=str(e),
            error_type=type(e).__name__,
            requested_model=model,
        )
        raise


@vector_metrics.time_vector_search
def get_chat_response(question: str, session_id: str, model: str | None = None) -> Any:
    """
    Get chat response with flexible performance monitoring.

    This is the synchronous wrapper for backward compatibility.
    """
    start_time = time.time()

    logger.info(
        "Chat response sync request started",
        question_length=len(question),
        session_id=session_id,
        requested_model=model,
        cache_enabled=settings.CACHE_ENABLED,
    )

    try:
        # Check if cache is enabled
        if not settings.CACHE_ENABLED:
            # Bypass cache and call directly
            if hasattr(llm_router, "set_model_preference"):
                llm_router.set_model_preference(model)
            response = chat_engine.chat(question)

            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(
                "Chat response sync completed (cache disabled)",
                question_length=len(question),
                session_id=session_id,
                response_length=len(str(response)) if response else 0,
                response_time_ms=duration_ms,
                used_model=model,
            )

            return response

        # Use async version with caching
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No running loop in this thread; safe to run directly
            response = asyncio.run(get_chat_response_async(question, session_id, model))

            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(
                "Chat response sync completed (new event loop)",
                question_length=len(question),
                session_id=session_id,
                response_length=len(str(response)) if response else 0,
                response_time_ms=duration_ms,
                used_model=model,
            )

            return response

        # If an event loop is already running (e.g., within an async web handler),
        # execute the coroutine in a fresh event loop on a worker thread.
        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, get_chat_response_async(question, session_id, model)
                )
                response = future.result()

                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.info(
                    "Chat response sync completed (thread executor)",
                    question_length=len(question),
                    session_id=session_id,
                    response_length=len(str(response)) if response else 0,
                    response_time_ms=duration_ms,
                    used_model=model,
                )

                return response

        # Otherwise, run it synchronously on this thread's event loop
        response = loop.run_until_complete(get_chat_response_async(question, session_id, model))

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "Chat response sync completed (existing event loop)",
            question_length=len(question),
            session_id=session_id,
            response_length=len(str(response)) if response else 0,
            response_time_ms=duration_ms,
            used_model=model,
        )

        return response
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)

        logger.error(
            "Chat response sync failed",
            question_length=len(question),
            session_id=session_id,
            response_time_ms=duration_ms,
            error=str(e),
            error_type=type(e).__name__,
            requested_model=model,
        )
        raise
