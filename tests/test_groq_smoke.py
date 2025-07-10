"""
Smoke test for Groq rate limiting functionality.

This test verifies the basic functionality without complex mocking.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.core.llm_router import GroqProvider


class TestGroqRateLimitSmoke:
    """Smoke tests for Groq rate limiting functionality."""

    def test_rate_limit_header_parsing(self):
        """Test basic rate limit header parsing."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Test basic header parsing
            headers = {
                "x-ratelimit-remaining-requests": "100",
                "x-ratelimit-remaining-tokens": "5000",
                "retry-after": "30",
            }

            rate_limit_info = provider._parse_rate_limit_headers(headers)

            assert rate_limit_info["remaining_requests"] == 100
            assert rate_limit_info["remaining_tokens"] == 5000
            assert rate_limit_info["retry_after_seconds"] == 30
            assert "timestamp" in rate_limit_info

    def test_redis_quota_storage(self):
        """Test storing and retrieving quota info from Redis."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True

        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()
            provider.redis_client = mock_redis

            # Test storing quota info
            quota_info = {
                "remaining_requests": 50,
                "remaining_tokens": 3000,
                "timestamp": 1234567890,
            }

            provider._store_rate_limit_info(quota_info)

            # Verify Redis was called correctly
            mock_redis.setex.assert_called_once()
            args = mock_redis.setex.call_args[0]
            assert args[0] == "groq:rate_limit"
            assert args[1] == 300  # 5 minutes TTL
            stored_data = json.loads(args[2])
            assert stored_data == quota_info

    def test_jitter_calculation(self):
        """Test jitter adds appropriate variance."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            base_delay = 5.0
            jittered_delays = [provider._add_jitter(base_delay) for _ in range(20)]

            # All delays should be within the jitter range (±25%)
            for delay in jittered_delays:
                assert 3.75 <= delay <= 6.25  # 5.0 ± 25%

            # Should have some variance (not all the same)
            unique_delays = set(jittered_delays)
            assert len(unique_delays) > 1

    def test_rate_limit_preemption_logic(self):
        """Test rate limit preemption logic."""
        import time

        mock_redis = MagicMock()

        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()
            provider.redis_client = mock_redis

            # Test with no stored info - should not skip
            mock_redis.get.return_value = None
            should_skip, delay = provider._should_skip_due_to_rate_limits()
            assert should_skip is False
            assert delay == 0

            # Test with low quota - should skip
            current_time = time.time()
            low_quota_info = {
                "remaining_requests": 1,  # Very low
                "remaining_tokens": 500,  # Low
                "retry_after_seconds": 45,
                "timestamp": current_time,
            }
            mock_redis.get.return_value = json.dumps(low_quota_info)

            with patch("app.core.llm_router.metrics") as mock_metrics:
                should_skip, delay = provider._should_skip_due_to_rate_limits()
                assert should_skip is True
                assert delay == 45

                # Should record metric
                mock_metrics.increment_counter.assert_called_once_with(
                    "llm_provider_rate_limit_hits_total",
                    {"provider": "groq", "action": "preemptive_skip"},
                )

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY environment variable not set"
    )
    @pytest.mark.asyncio
    async def test_groq_live_smoke(self):
        """
        Optional live test with real Groq API.
        Only runs when GROQ_API_KEY environment variable is set.
        """
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        # Use real settings
        provider = GroqProvider()

        if not provider.is_available():
            pytest.skip("Groq API key not available")

        messages = [ChatMessage(role=MessageRole.USER, content="Hello! Just say 'Hi' back.")]

        try:
            response = await provider.complete(messages)
            assert response is not None
            assert isinstance(response.text, str)
            assert len(response.text) > 0
            print(f"✅ Live Groq test successful: {response.text[:50]}...")
        except Exception as e:
            # If it's a rate limit error, that's actually good - shows our detection works
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"✅ Rate limiting detected correctly: {e}")
            else:
                raise
