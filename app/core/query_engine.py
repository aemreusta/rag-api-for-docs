from langfuse.client import Langfuse
from llama_index import ServiceContext, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores import PGVectorStore

from app.core.config import settings


class QueryEngine:
    def __init__(self):
        # Initialize Langfuse for tracking
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        # Setup vector store
        self.vector_store = PGVectorStore.from_params(
            dsn=str(settings.DATABASE_URL),
            table_name="content_embeddings",
        )

        # Setup LlamaIndex components
        self.embed_model = OpenAIEmbedding()
        self.llm = OpenRouter(api_key=settings.OPENROUTER_API_KEY, model=settings.LLM_MODEL_NAME)

        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )

        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        # Create a ChatEngine for conversational context
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            retriever=self.index.as_retriever(),
            llm=self.llm,
        )

    async def query(self, query_text: str, chat_history: list | None = None) -> str:
        # Create a Langfuse trace for monitoring
        trace = self.langfuse.trace(name="query")

        try:
            # Execute query and track with Langfuse
            with trace.span(name="query_execution"):
                response = self.chat_engine.chat(query_text)

            trace.end()
            return str(response)

        except Exception as e:
            trace.error(error=str(e))
            trace.end()
            raise e


# Create a global instance
query_engine = QueryEngine()


def get_chat_response(question: str, session_id: str):
    # NOTE: For now, session_id is a placeholder. Real memory will be added in Phase 3.
    # This engine will internally condense the question but doesn't have persistent memory yet.
    response = query_engine.chat_engine.chat(question)
    return response
