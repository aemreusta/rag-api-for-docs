"""
Dimension-specific embedding models for handling PostgreSQL pgvector HNSW limitations.

Each model represents a table optimized for specific embedding dimensions:
- EmbeddingTable768: HuggingFace and small models (384-768 dims)
- EmbeddingTable1024: Qwen3-Embedding-0.6B (1024 dims)
- EmbeddingTable1536: Google Gemini MRL (1536 dims)
- EmbeddingTable3072: Google Gemini full (3072 dims - uses IVFFlat if needed)
"""

from datetime import datetime

from sqlalchemy import Column, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import declarative_base

from app.db.models import Vector

Base = declarative_base()


class BaseEmbeddingTable:
    """Base class for dimension-specific embedding tables."""

    id = Column(Integer, primary_key=True, index=True)
    source_document = Column(String(255), nullable=False, index=True)
    page_number = Column(Integer, nullable=False, index=True)
    content_text = Column(Text, nullable=False)
    provider = Column(String(50), nullable=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class EmbeddingTable768(BaseEmbeddingTable, Base):
    """Embedding table for 768-dimensional vectors (HuggingFace models)."""

    __tablename__ = "embeddings_768"

    content_vector = Column(Vector(768), nullable=True)

    __table_args__ = (
        Index("idx_embeddings_768_source_document", "source_document"),
        Index("idx_embeddings_768_page_number", "page_number"),
        Index("idx_embeddings_768_provider", "provider"),
        # HNSW index for vector similarity (created in migration)
    )


class EmbeddingTable1024(BaseEmbeddingTable, Base):
    """Embedding table for 1024-dimensional vectors (Qwen models)."""

    __tablename__ = "embeddings_1024"

    content_vector = Column(Vector(1024), nullable=True)

    __table_args__ = (
        Index("idx_embeddings_1024_source_document", "source_document"),
        Index("idx_embeddings_1024_page_number", "page_number"),
        Index("idx_embeddings_1024_provider", "provider"),
        # HNSW index for vector similarity (created in migration)
    )


class EmbeddingTable1536(BaseEmbeddingTable, Base):
    """Embedding table for 1536-dimensional vectors (Google Gemini MRL)."""

    __tablename__ = "embeddings_1536"

    content_vector = Column(Vector(1536), nullable=True)

    __table_args__ = (
        Index("idx_embeddings_1536_source_document", "source_document"),
        Index("idx_embeddings_1536_page_number", "page_number"),
        Index("idx_embeddings_1536_provider", "provider"),
        # HNSW index for vector similarity (created in migration)
    )


class EmbeddingTable3072(BaseEmbeddingTable, Base):
    """Embedding table for 3072-dimensional vectors (Google Gemini full)."""

    __tablename__ = "embeddings_3072"

    content_vector = Column(Vector(3072), nullable=True)

    __table_args__ = (
        Index("idx_embeddings_3072_source_document", "source_document"),
        Index("idx_embeddings_3072_page_number", "page_number"),
        Index("idx_embeddings_3072_provider", "provider"),
        # Note: Will use IVFFlat index due to HNSW 2000-dim limit
    )


# Dimension to table model mapping
DIMENSION_TABLE_MAP = {
    768: EmbeddingTable768,
    1024: EmbeddingTable1024,
    1536: EmbeddingTable1536,
    3072: EmbeddingTable3072,
}

# Provider to dimension mapping
PROVIDER_DIMENSION_MAP = {
    "huggingface": 768,
    "qwen": 1024,
    "google": 3072,  # Full dimensions by default
    "google_mrl": 1536,  # Matryoshka Representation Learning
}


def get_embedding_table_for_dimension(dimension: int) -> type[BaseEmbeddingTable]:
    """Get the appropriate embedding table class for given dimension."""
    if dimension not in DIMENSION_TABLE_MAP:
        # Find the closest supported dimension
        supported_dims = sorted(DIMENSION_TABLE_MAP.keys())
        closest_dim = min(supported_dims, key=lambda x: abs(x - dimension))
        return DIMENSION_TABLE_MAP[closest_dim]

    return DIMENSION_TABLE_MAP[dimension]


def get_embedding_table_for_provider(provider: str) -> type[BaseEmbeddingTable]:
    """Get the appropriate embedding table class for given provider."""
    dimension = PROVIDER_DIMENSION_MAP.get(provider, 768)  # Default to 768
    return get_embedding_table_for_dimension(dimension)
