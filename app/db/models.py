from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship
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


# Ingestion schema models


class DocumentStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    content_hash = Column(String(64), nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    page_count = Column(Integer)
    word_count = Column(Integer)
    language = Column(String(10), default="en")
    uploaded_at = Column(DateTime)
    updated_at = Column(DateTime)
    processed_at = Column(DateTime)
    author = Column(String(100))
    tags = Column(ARRAY(String))
    status = Column(
        SAEnum(DocumentStatusEnum, name="document_status"), default=DocumentStatusEnum.PENDING
    )
    version = Column(Integer, default=1)
    storage_backend = Column(String(20), default="local")
    storage_uri = Column(Text, nullable=False)
    soft_deleted = Column(Boolean, default=False)
    created_by = Column(String(100))
    extra_metadata = Column(JSON, default={})

    __table_args__ = (
        CheckConstraint("file_size > 0 AND file_size <= 104857600", name="valid_file_size"),
        CheckConstraint("version > 0", name="valid_version"),
        Index("idx_documents_content_hash", "content_hash"),
        Index("idx_documents_status", "status", postgresql_where=(~(soft_deleted))),
        Index("idx_documents_tags", "tags", postgresql_using="gin"),
        Index("idx_documents_uploaded_at", "uploaded_at"),
    )

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    token_count = Column(Integer)
    chunk_type = Column(String(50), default="paragraph")
    page_number = Column(Integer)
    section_title = Column(String(255))
    extra_metadata = Column(JSON, default={})
    created_at = Column(DateTime)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("uq_document_chunk_idx", "document_id", "chunk_index", unique=True),
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_content_hash", "content_hash"),
        CheckConstraint("chunk_index >= 0", name="valid_chunk_index"),
        CheckConstraint("token_count IS NULL OR token_count > 0", name="valid_token_count"),
    )


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String, primary_key=True)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), default="pending")
    input_data = Column(JSON, nullable=False)
    result_data = Column(JSON)
    error_message = Column(Text)
    progress_percent = Column(Integer, default=0)
    created_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_by = Column(String(100))

    __table_args__ = (
        CheckConstraint("progress_percent >= 0 AND progress_percent <= 100", name="valid_progress"),
        Index("idx_processing_jobs_status", "status", "created_at"),
    )


class QueryLog(Base):
    """
    Table for logging user queries and responses for analytics.
    """

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    session_id = Column(String(255))
