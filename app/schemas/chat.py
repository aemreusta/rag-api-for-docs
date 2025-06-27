from pydantic import BaseModel, Field


class SourceNode(BaseModel):
    """Source node information from retrieved documents"""

    text: str
    source: str
    page_number: int | None = None
    score: float | None = None

    @classmethod
    def from_llama_index(cls, node):
        """Convert LlamaIndex source node to Pydantic model"""
        return cls(
            text=node.text,
            source=node.metadata.get("source_document", "unknown"),
            page_number=node.metadata.get("page_number"),
            score=getattr(node, "score", None),
        )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the chatbot")
    session_id: str = Field(
        ..., min_length=1, description="Session identifier for conversation context"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceNode] = []
