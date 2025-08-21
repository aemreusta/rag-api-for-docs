"""
Optional smoke tests for Google AI Studio provider.

These run only when GOOGLE_AI_STUDIO_API_KEY is set.
"""

import os

import pytest

from app.core.llm_router import GoogleAIStudioProvider


@pytest.mark.skipif(
    not os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
    reason="GOOGLE_AI_STUDIO_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_google_live_smoke():
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    provider = GoogleAIStudioProvider()
    if not provider.is_available():
        pytest.skip("Google Gemini not available or client not installed")

    messages = [ChatMessage(role=MessageRole.USER, content="Say 'Hi'.")]
    response = await provider.complete(messages)
    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
