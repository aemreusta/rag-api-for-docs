from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.redis import redis_client
from app.main import app

client = TestClient(app)


def test_chat_endpoint_without_api_key():
    """Auth disabled: endpoint should work without API key."""
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the volunteer policy?",
            "session_id": "test-session",
            "stream": False,
        },
    )
    # Either 200 on happy path or 500 friendly error on internal failure
    assert response.status_code in (200, 500)


@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
def test_chat_endpoint_with_invalid_api_key(mock_rag_async):
    """Auth disabled: invalid key should be ignored."""
    # Minimal async RAG mock to avoid un-awaited coroutine warnings
    mock_rag_async.return_value = MagicMock(__str__=lambda _: "OK", source_nodes=[])
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the volunteer policy?",
            "session_id": "test-session",
            "stream": False,
        },
        headers={"X-API-Key": "invalid-key"},
    )
    assert response.status_code in (200, 500)


@patch("app.api.v1.chat.get_chat_response")
@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
@patch("app.api.v1.chat.langfuse_client")
def test_chat_endpoint_successful_response(mock_langfuse, mock_rag_async, mock_get_chat_response):
    """Test successful chat endpoint response."""
    # Mock Langfuse
    mock_trace = MagicMock()
    mock_generation = MagicMock()
    mock_trace.generation.return_value = mock_generation
    mock_langfuse.trace.return_value = mock_trace

    # Mock chat response
    mock_response = MagicMock()
    mock_response.__str__ = lambda x: "Volunteers must complete training before starting."
    mock_response.source_nodes = [
        MagicMock(
            text="Volunteer training is required",
            metadata={"source_document": "policy.pdf", "page_number": 1},
            score=0.95,
        )
    ]
    mock_get_chat_response.return_value = mock_response
    mock_rag_async.return_value = mock_response

    # Make request
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the volunteer policy?",
            "session_id": "test-session",
            "stream": False,
        },
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["text"] == "Volunteer training is required"

    # Verify Langfuse was called
    mock_langfuse.trace.assert_called_once()
    mock_generation.end.assert_called_once()


@patch("app.api.v1.chat.get_chat_response")
@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
@patch("app.api.v1.chat.langfuse_client")
def test_chat_endpoint_handles_errors(mock_langfuse, mock_rag_async, mock_get_chat_response):
    """Test that chat endpoint handles errors gracefully."""
    # Mock Langfuse
    mock_trace = MagicMock()
    mock_generation = MagicMock()
    mock_trace.generation.return_value = mock_generation
    mock_langfuse.trace.return_value = mock_trace

    # Mock error in chat response
    mock_get_chat_response.side_effect = Exception("Test error")
    mock_rag_async.side_effect = Exception("Test error")

    # Make request
    response = client.post(
        "/api/v1/chat",
        json={"question": "What is the volunteer policy?", "session_id": "test-session"},
        headers={"X-API-Key": settings.API_KEY},
    )

    # Verify error response (friendly generic message)
    assert response.status_code == 500
    assert "Üzgünüm" in response.json()["detail"]

    # Verify Langfuse error was logged
    mock_generation.end.assert_called_with(level="ERROR", status_message="Test error")


def test_chat_request_validation():
    """Test request validation for required fields."""
    # Missing question
    response = client.post("/api/v1/chat", json={"session_id": "test-session", "stream": False})
    assert response.status_code == 422

    # Missing session_id
    response = client.post("/api/v1/chat", json={"question": "Test question", "stream": False})
    assert response.status_code == 422

    # Empty question
    response = client.post(
        "/api/v1/chat",
        json={"question": "", "session_id": "test-session", "stream": False},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "question,session_id",
    [
        ("What is the policy?", "session-123"),
        ("How do I volunteer?", "user-456"),
        ("Tell me about training requirements", "guest-789"),
    ],
)
@patch("app.api.v1.chat.get_chat_response")
@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
@patch("app.api.v1.chat.langfuse_client")
def test_chat_endpoint_various_inputs(
    mock_langfuse, mock_rag_async, mock_get_chat_response, question, session_id
):
    """Test chat endpoint with various valid inputs."""
    # Mock response
    mock_langfuse.trace.return_value.generation.return_value = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda x: f"Answer to: {question}"
    mock_response.source_nodes = []
    mock_get_chat_response.return_value = mock_response
    mock_rag_async.return_value = mock_response

    response = client.post(
        "/api/v1/chat",
        json={"question": question, "session_id": session_id, "stream": False},
    )

    assert response.status_code == 200
    assert question.lower() in response.json()["answer"].lower()


@pytest_asyncio.fixture
async def redis_cleaner():
    """Fixture to clean Redis before and after the test."""
    await redis_client.flushdb()
    yield
    await redis_client.flushdb()


@pytest.mark.asyncio
@patch("app.api.v1.chat.get_chat_response")
@patch("app.api.v1.chat.get_chat_response_async", new_callable=AsyncMock)
@patch("app.api.v1.chat.langfuse_client")
async def test_chat_endpoint_rate_limiting(
    mock_langfuse, mock_rag_async, mock_get_chat_response, redis_cleaner
):
    """Test that rate limiting is applied to the chat endpoint."""
    # Import here to get the actual rate limiting function
    from app.api.deps import rate_limit
    from app.core.ratelimit import RateLimiter

    # Mock the successful response pathway
    mock_response = MagicMock()
    mock_response.__str__ = lambda x: "This is a test answer."
    mock_response.source_nodes = []
    mock_get_chat_response.return_value = mock_response
    mock_rag_async.return_value = mock_response
    mock_langfuse.trace.return_value.generation.return_value = MagicMock()

    # Create a test rate limiting function with a low limit
    async def test_rate_limit(request: Request):
        limiter = RateLimiter(redis_client, limit=2, window=60)
        await limiter.check(request)

    # Use FastAPI's dependency_overrides to replace the rate limiting dependency
    app.dependency_overrides[rate_limit] = test_rate_limit

    try:
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            headers = {}
            json_payload = {
                "question": "What is the volunteer policy?",
                "session_id": "test-session-rate-limit",
                "stream": False,
            }

            # First two requests should succeed
            for _ in range(2):
                response = await ac.post("/api/v1/chat", json=json_payload, headers=headers)
                assert response.status_code == 200

            # Third request should be rate-limited
            response = await ac.post("/api/v1/chat", json=json_payload, headers=headers)
            assert response.status_code == 429
            assert "Too many requests" in response.json()["detail"]
            assert "Retry-After" in response.headers
    finally:
        # Clean up dependency overrides to avoid test interference
        app.dependency_overrides.clear()
