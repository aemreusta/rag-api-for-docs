from typing import Optional

import chromadb
from langfuse.client import Langfuse
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore

from app.core.config import settings


class QueryEngine:
    def __init__(self):
        # Initialize Langfuse for tracking
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_or_create_collection("documents")

        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Setup LlamaIndex components
        self.embed_model = OpenAIEmbedding()
        self.llm = OpenAI(temperature=0.1)

        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )

        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            service_context=self.service_context,
        )

    async def query(self, query_text: str, chat_history: Optional[list] = None) -> str:
        # Create a Langfuse trace for monitoring
        trace = self.langfuse.trace(name="query")

        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                streaming=False,
                similarity_top_k=3,
            )

            # Execute query and track with Langfuse
            with trace.span(name="query_execution"):
                response = query_engine.query(query_text)

            trace.end()
            return str(response)

        except Exception as e:
            trace.error(error=str(e))
            trace.end()
            raise e


# Create a global instance
query_engine = QueryEngine()
