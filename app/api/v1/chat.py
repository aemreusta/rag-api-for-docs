import langfuse
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_api_key
from app.core.config import settings
from app.core.query_engine import get_chat_response
from app.schemas.chat import ChatRequest, ChatResponse, SourceNode

router = APIRouter()
langfuse_client = langfuse.Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
def handle_chat(request: ChatRequest):
    trace = langfuse_client.trace(name="chat-request", user_id=request.session_id)
    generation = trace.generation(name="rag-response", input=request.question)

    try:
        rag_response = get_chat_response(request.question, request.session_id)

        sources = [SourceNode.from_llama_index(node) for node in rag_response.source_nodes]
        response = ChatResponse(answer=str(rag_response), sources=sources)

        generation.end(output=response.model_dump())
        return response
    except Exception as e:
        generation.end(level="ERROR", status_message=str(e))
        raise HTTPException(status_code=500, detail="An error occurred.") from e
