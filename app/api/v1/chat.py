import time

import langfuse
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_api_key, rate_limit
from app.core.config import settings
from app.core.logging_config import get_logger, get_trace_id
from app.core.query_engine import get_chat_response
from app.schemas.chat import ChatRequest, ChatResponse, SourceNode

router = APIRouter()
logger = get_logger(__name__)

langfuse_client = langfuse.Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)


@router.post(
    "/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key), Depends(rate_limit)]
)
def handle_chat(request: ChatRequest):
    """
    Handle chat requests with structured logging and trace correlation.
    """
    start_time = time.time()

    # Get trace ID from logging context for correlation with Langfuse
    trace_id = get_trace_id() or "unknown"

    logger.info(
        "Chat request received",
        session_id=request.session_id,
        question_length=len(request.question),
        trace_id=trace_id,
    )

    # Create Langfuse trace with the same trace_id for correlation
    trace = langfuse_client.trace(
        name="chat-request",
        user_id=request.session_id,
        id=trace_id,  # Use the same trace_id from logging context
    )
    generation = trace.generation(name="rag-response", input=request.question)

    try:
        logger.info(
            "Calling RAG engine",
            session_id=request.session_id,
            model=settings.LLM_MODEL_NAME,
            trace_id=trace_id,
        )

        rag_response = get_chat_response(request.question, request.session_id)

        sources = [SourceNode.from_llama_index(node) for node in rag_response.source_nodes]
        response = ChatResponse(answer=str(rag_response), sources=sources)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        logger.info(
            "Chat request completed successfully",
            session_id=request.session_id,
            processing_time_ms=round(processing_time, 2),
            answer_length=len(response.answer),
            source_count=len(sources),
            trace_id=trace_id,
        )

        generation.end(output=response.model_dump())
        return response

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000

        logger.error(
            "Chat request failed",
            session_id=request.session_id,
            processing_time_ms=round(processing_time, 2),
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id,
            exc_info=True,
        )

        generation.end(level="ERROR", status_message=str(e))
        # Return a generic, user-friendly error message without leaking internals
        raise HTTPException(
            status_code=500,
            detail=(
                "Üzgünüm, şu anda yanıt veremiyorum. Lütfen kısa bir süre sonra tekrar deneyin."
            ),
        ) from e
