from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class ContentEmbedding(Base):
    """
    Table for storing document content and embeddings.
    This represents the charity_policies table that LlamaIndex creates.
    """

    __tablename__ = "charity_policies"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text)  # JSON metadata about the document
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class QueryLog(Base):
    """
    Table for logging user queries and responses for analytics.
    """

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    session_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
