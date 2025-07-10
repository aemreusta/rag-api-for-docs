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
from typing import Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.cache import cache_key_for_chat, cached
from app.core.config import settings
from app.core.llm_router import LLMRouter
from app.core.metrics import vector_metrics

# Set up the embedding model first
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Initialize components
vector_store = PGVectorStore.from_params(
    database=settings.POSTGRES_DB,
    host=settings.POSTGRES_SERVER,
    password=settings.POSTGRES_PASSWORD,
    port=5432,
    user=settings.POSTGRES_USER,
    table_name="content_embeddings",
    embed_dim=settings.EMBEDDING_DIM,  # Use configurable dimension from settings
)
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


def _cache_key_for_chat_response(question: str, session_id: str) -> str:
    """Generate cache key for chat responses using the LLM model name."""
    return cache_key_for_chat(
        question,
        session_id,
        llm_router._available_providers[0].get_model_name()
        if llm_router._available_providers
        else "",
    )


@vector_metrics.time_vector_search
@cached(ttl=settings.CACHE_TTL_SECONDS, key_func=_cache_key_for_chat_response)
async def get_chat_response_async(question: str, session_id: str) -> Any:
    """Get chat response with caching and flexible performance monitoring."""
    # NOTE: For now, session_id is a placeholder. Real memory will be added in Phase 3.
    # This engine will internally condense the question but doesn't have persistent memory yet.

    # Run the chat in executor since LlamaIndex chat_engine.chat is synchronous
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, chat_engine.chat, question)
    return response


@vector_metrics.time_vector_search
def get_chat_response(question: str, session_id: str) -> Any:
    """
    Get chat response with flexible performance monitoring.

    This is the synchronous wrapper for backward compatibility.
    """
    # Check if cache is enabled
    if not settings.CACHE_ENABLED:
        # Bypass cache and call directly
        response = chat_engine.chat(question)
        return response

    # Use async version with caching
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(get_chat_response_async(question, session_id))
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(get_chat_response_async(question, session_id))
        finally:
            loop.close()
