from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings

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
llm = OpenRouter(api_key=settings.OPENROUTER_API_KEY, model=settings.LLM_MODEL_NAME)

# Create a query engine first
query_engine = index.as_query_engine(llm=llm)

# Create a ChatEngine for conversational context
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    llm=llm,
)


def get_chat_response(question: str, session_id: str):
    # NOTE: For now, session_id is a placeholder. Real memory will be added in Phase 3.
    # This engine will internally condense the question but doesn't have persistent memory yet.
    response = chat_engine.chat(question)
    return response
