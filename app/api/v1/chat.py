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


async def _call_rag_engine(question: str, session_id: str):
    """Call RAG engine preferring async path with sync fallback."""
    try:
        return await get_chat_response_async(question, session_id)
    except Exception:
        return get_chat_response(question, session_id)


def _extract_answer_and_sources(rag_response) -> tuple[str, list[SourceNode]]:
    """Extract primary answer text and sources from a RAG response."""
    raw_answer = str(rag_response) if rag_response is not None else ""
    sources = [
        SourceNode.from_llama_index(node) for node in getattr(rag_response, "source_nodes", [])
    ]
    return raw_answer, sources


def _is_empty_answer(text: str) -> bool:
    """Determine if an answer is empty or placeholder."""
    return not text.strip() or text.strip().lower() == "empty response"


async def _llm_direct_answer(question: str) -> str | None:
    """Direct LLM fallback when RAG returns empty answer."""
    try:
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        llm_result = await llm_router.acomplete(
            [ChatMessage(role=MessageRole.USER, content=question)]
        )
        return getattr(llm_result, "text", None) or str(llm_result)
    except Exception as llm_err:  # pragma: no cover - logging path
        logger.warning(
            "LLM fallback failed",
            error=str(llm_err),
            error_type=type(llm_err).__name__,
        )
        return None


def _create_streaming_response(question: str, answer_text: str | None) -> StreamingResponse | None:
    """Create streaming response using provider streaming with chunked fallback.

    Returns a StreamingResponse if streaming is possible/appropriate; otherwise None.
    """
    # Try true provider streaming first
    try:
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        async def provider_streamer():
            async for chunk in llm_router.astream_complete(
                [ChatMessage(role=MessageRole.USER, content=question)]
            ):
                text = getattr(chunk, "text", None) or getattr(chunk, "delta", None) or ""
                if text:
                    yield text

        return StreamingResponse(provider_streamer(), media_type="text/plain; charset=utf-8")
    except Exception:
        # Fallback to simple chunked streaming for long answers
        if answer_text and len(answer_text) > 300:

            async def streamer():
                chunk_size = 256
                text = answer_text
                for i in range(0, len(text), chunk_size):
                    yield text[i : i + chunk_size]

            return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")

    return None


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

        rag_response = await _call_rag_engine(request.question, request.session_id)
        raw_answer, sources = _extract_answer_and_sources(rag_response)

        # Fallback to direct LLM if RAG returned empty/placeholder
        if _is_empty_answer(raw_answer):
            llm_text = await _llm_direct_answer(request.question)
            if llm_text and llm_text.strip():
                response = ChatResponse(answer=llm_text, sources=[])
                generation.end(output=response.model_dump())
                return response

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

        if request.stream:
            streaming = _create_streaming_response(request.question, response.answer)
            if streaming is not None:
                return streaming

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
