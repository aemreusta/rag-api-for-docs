from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import UserDefinedType

from app.core.config import settings

Base = declarative_base()


class Vector(UserDefinedType):
    """Custom SQLAlchemy type for PostgreSQL pgvector extension."""

    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def get_col_spec(self):
        if self.dimensions:
            return f"VECTOR({self.dimensions})"
        return "VECTOR"


class ContentEmbedding(Base):
    """
    Table for storing document content and embeddings for RAG pipeline.
    This stores indexed text chunks from source PDF documents and their vector embeddings.
    """

    __tablename__ = "content_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    source_document = Column(String(255), nullable=False, index=True)  # PDF filename for citations
    page_number = Column(Integer, nullable=False, index=True)  # Page number for citations
    content_text = Column(Text, nullable=False)  # Actual text content shown to LLM
    content_vector = Column(
        Vector(settings.EMBEDDING_DIM), nullable=True
    )  # Vector embeddings for similarity search (configurable, default 384)


class QueryLog(Base):
    """
    Table for logging user queries and responses for analytics.
    """

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    session_id = Column(String(255))
