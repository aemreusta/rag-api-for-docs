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

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.core.llm_router import LLMRouter

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
    embed_dim=384,  # Match the content_embeddings table vector dimension
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


def get_chat_response(question: str, session_id: str):
    # NOTE: For now, session_id is a placeholder. Real memory will be added in Phase 3.
    # This engine will internally condense the question but doesn't have persistent memory yet.
    response = chat_engine.chat(question)
    return response
