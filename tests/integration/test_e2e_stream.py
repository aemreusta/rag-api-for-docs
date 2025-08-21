from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
@patch("app.api.v1.chat.langfuse_client")
@patch("app.api.v1.chat.llm_router")
@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
async def test_chat_streaming_e2e(mock_rag_async, mock_llm_router, mock_langfuse):
    # Mock Langfuse objects
    mock_trace = MagicMock()
    mock_generation = MagicMock()
    mock_trace.generation.return_value = mock_generation
    mock_langfuse.trace.return_value = mock_trace

    # Prepare a mocked streaming generator from router
    async def fake_stream():
        for chunk in ["Hello ", "world", "! This ", "is streamed."]:
            fake_resp = MagicMock()
            fake_resp.text = chunk
            yield fake_resp

    # router.astream_complete is iterated with `async for`, so it must return an async iterator
    mock_llm_router.astream_complete.return_value = fake_stream()
    mock_rag_async.return_value = MagicMock(__str__=lambda self: "RAG placeholder", source_nodes=[])

    # Make streaming request
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"question": "long question?", "session_id": "s1", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        collected = "".join([part.decode("utf-8") for part in resp.iter_bytes()])
        assert "Hello world! This is streamed.".replace(" ", "") in collected.replace(" ", "")

    mock_generation.end.assert_called()


@pytest.mark.asyncio
@patch("app.api.v1.chat.langfuse_client")
@patch("app.api.v1.chat.llm_router")
@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
async def test_chat_chunked_streaming_fallback(mock_rag_async, mock_llm_router, mock_langfuse):
    # Mock Langfuse objects
    mock_trace = MagicMock()
    mock_generation = MagicMock()
    mock_trace.generation.return_value = mock_generation
    mock_langfuse.trace.return_value = mock_trace

    # Force provider streaming to fail to trigger chunked fallback
    async def failing_stream():
        raise RuntimeError("provider streaming failed")

    mock_llm_router.astream_complete.side_effect = RuntimeError("provider streaming failed")

    # Prepare long RAG answer to trigger should_stream based on length
    long_answer = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
    rag_resp = MagicMock()
    rag_resp.__str__ = lambda self: long_answer
    rag_resp.source_nodes = []
    mock_rag_async.return_value = rag_resp

    # Make request without explicit stream flag; length-based streaming should kick in
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"question": "long?", "session_id": "s2"},
    ) as resp:
        assert resp.status_code == 200
        # Should be plain text stream via chunked fallback
        assert resp.headers.get("content-type", "").startswith("text/plain")
        collected = b"".join(list(resp.iter_bytes())).decode("utf-8")
        assert long_answer[:100] in collected
        assert len(collected) >= len(long_answer) - 10

    mock_generation.end.assert_called()
