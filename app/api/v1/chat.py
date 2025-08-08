import time

import langfuse
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import rate_limit
from app.core.config import settings
from app.core.logging_config import get_logger, get_trace_id
from app.core.query_engine import get_chat_response, get_chat_response_async, llm_router
from app.schemas.chat import ChatRequest, ChatResponse, SourceNode

router = APIRouter()
logger = get_logger(__name__)

langfuse_client = langfuse.Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest, _rl: None = Depends(rate_limit)):
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

        # Prefer async path to avoid event loop mixing
        try:
            rag_response = await get_chat_response_async(request.question, request.session_id)
        except Exception:
            # Fallback to sync path (uses cached wrapper) if needed
            rag_response = get_chat_response(request.question, request.session_id)

        # Primary answer from RAG
        raw_answer = str(rag_response) if rag_response is not None else ""
        sources = [
            SourceNode.from_llama_index(node) for node in getattr(rag_response, "source_nodes", [])
        ]

        # If underlying RAG returns an empty placeholder, fall back to direct LLM
        if not raw_answer.strip() or raw_answer.strip().lower() == "empty response":
            try:
                from llama_index.core.base.llms.types import ChatMessage, MessageRole

                llm_result = await llm_router.acomplete(
                    [ChatMessage(role=MessageRole.USER, content=request.question)]
                )
                llm_text = getattr(llm_result, "text", None) or str(llm_result)
                if llm_text and llm_text.strip():
                    response = ChatResponse(answer=llm_text, sources=[])
                    generation.end(output=response.model_dump())
                    return response
            except Exception as llm_err:
                logger.warning(
                    "LLM fallback failed",
                    error=str(llm_err),
                    error_type=type(llm_err).__name__,
                    trace_id=trace_id,
                )
            # As a last resort, raise a specific RAG-empty error that clients can detect
            generation.end(level="ERROR", status_message="RAG returned empty answer")
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "RAG_EMPTY",
                    "message": "RAG engine produced no answer and fallback failed.",
                    "trace_id": trace_id,
                },
            )

        response = ChatResponse(answer=raw_answer, sources=sources)

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

        # If streaming requested and answer is long, stream in chunks
        if request.stream and len(response.answer) > 300:

            async def streamer():
                chunk_size = 256
                text = response.answer
                for i in range(0, len(text), chunk_size):
                    yield text[i : i + chunk_size]

            return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")

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
        # Preserve existing test expectations: return 500 with friendly message
        msg = "Üzgünüm, şu anda yanıt veremiyorum. Lütfen kısa bir süre sonra tekrar deneyin."
        raise HTTPException(status_code=500, detail=msg) from None
