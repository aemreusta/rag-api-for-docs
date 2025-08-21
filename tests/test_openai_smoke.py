"""
Optional smoke tests for OpenAI provider. These run only when OPENAI_API_KEY is set.
"""

import os

import pytest

from app.core.llm_router import OpenAIProvider


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY environment variable not set"
)
@pytest.mark.asyncio
async def test_openai_live_smoke():
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    provider = OpenAIProvider()
    if not provider.is_available():
        pytest.skip("OpenAI API key not available or client not installed")

    messages = [ChatMessage(role=MessageRole.USER, content="Say 'Hi'.")]
    response = await provider.complete(messages)
    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0
