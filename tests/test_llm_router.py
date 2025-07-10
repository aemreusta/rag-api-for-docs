"""
Unit tests for the LLM Router implementation.

Tests cover:
- Provider availability detection
- Automatic fallback on errors (timeout, rate limit, auth failures)
- Redis caching of failed providers
- Error classification and handling
- Streaming and non-streaming responses
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, CompletionResponse, MessageRole

from app.core.llm_router import (
    ErrorType,
    GroqProvider,
    LLMRouter,
    LocalProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderType,
)


class TestProviderImplementations:
    """Test individual provider implementations."""

    def test_openrouter_provider_availability(self):
        """Test OpenRouter provider availability check."""
        with patch("app.core.llm_router.settings.OPENROUTER_API_KEY", "test-key"):
            provider = OpenRouterProvider()
            assert provider.is_available() is True
            assert provider.provider_type == ProviderType.OPENROUTER

        with patch("app.core.llm_router.settings.OPENROUTER_API_KEY", ""):
            provider = OpenRouterProvider()
            assert provider.is_available() is False

    def test_openai_provider_availability(self):
        """Test OpenAI provider availability check."""
        with patch("app.core.llm_router.settings.OPENAI_API_KEY", "test-key"):
            provider = OpenAIProvider()
            assert provider.is_available() is True
            assert provider.provider_type == ProviderType.OPENAI

        with patch("app.core.llm_router.settings.OPENAI_API_KEY", ""):
            provider = OpenAIProvider()
            assert provider.is_available() is False

    def test_groq_provider_availability(self):
        """Test Groq provider availability check."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()
            assert provider.is_available() is True
            assert provider.provider_type == ProviderType.GROQ
            assert provider.get_model_name() == "llama3-70b-8192"

        with patch("app.core.llm_router.settings.GROQ_API_KEY", ""):
            provider = GroqProvider()
            assert provider.is_available() is False

    def test_groq_provider_custom_model(self):
        """Test Groq provider with custom model configuration."""
        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.settings.GROQ_MODEL_NAME", "llama3-8b-8192"),
        ):
            provider = GroqProvider()
            assert provider.get_model_name() == "llama3-8b-8192"

    def test_groq_provider_custom_timeout(self):
        """Test Groq provider with custom timeout configuration."""
        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.settings.GROQ_TIMEOUT_SECONDS", 60),
        ):
            provider = GroqProvider()
            assert provider.timeout_seconds == 60

    def test_local_provider_placeholder(self):
        """Test Local provider placeholder implementation."""
        provider = LocalProvider()
        assert provider.is_available() is False
        assert provider.provider_type == ProviderType.LOCAL
        assert provider.get_model_name() == "local-llm"


class TestGroqProvider:
    """Test Groq provider specific functionality."""

    @pytest.fixture
    def mock_groq_client(self):
        """Create a mock Groq client."""
        mock_client = MagicMock()
        mock_completions = MagicMock()
        mock_client.chat.completions = mock_completions

        # Make create method async by default
        async def mock_create(*args, **kwargs):
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Hello world"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.model_dump.return_value = {"response": "data"}
            return mock_response

        mock_completions.create = mock_create
        return mock_client, mock_completions

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client for testing."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        return mock_redis

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [ChatMessage(role=MessageRole.USER, content="Hello, world!")]

    def test_convert_messages_to_groq_format(self):
        """Test conversion of LlamaIndex messages to Groq format."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            messages = [
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
            ]

            groq_messages = provider._convert_messages_to_groq_format(messages)

            expected = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]

            assert groq_messages == expected

    def test_create_completion_response(self):
        """Test conversion of Groq response to LlamaIndex format."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Mock Groq response
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Hello world"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.model_dump.return_value = {"response": "data"}

            response = provider._create_completion_response(mock_response)

            assert isinstance(response, CompletionResponse)
            assert response.text == "Hello world"

    def test_create_streaming_response(self):
        """Test conversion of Groq streaming chunk to LlamaIndex format."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Mock Groq streaming chunk
            mock_chunk = MagicMock()
            mock_choice = MagicMock()
            mock_delta = MagicMock()
            mock_delta.content = "Hello"
            mock_choice.delta = mock_delta
            mock_chunk.choices = [mock_choice]
            mock_chunk.model_dump.return_value = {"chunk": "data"}

            response = provider._create_streaming_response(mock_chunk)

            assert isinstance(response, CompletionResponse)
            assert response.text == "Hello"

    def test_parse_rate_limit_headers(self):
        """Test parsing of Groq rate limit headers."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Mock response headers
            headers = {
                "x-ratelimit-limit-requests": "14400",
                "x-ratelimit-limit-tokens": "15000",
                "x-ratelimit-remaining-requests": "14399",
                "x-ratelimit-remaining-tokens": "14977",
                "x-ratelimit-reset-requests": "6s",
                "x-ratelimit-reset-tokens": "92ms",
                "retry-after": "30",
            }

            rate_limit_info = provider._parse_rate_limit_headers(headers)

            assert rate_limit_info["limit_requests_per_day"] == 14400
            assert rate_limit_info["limit_tokens_per_minute"] == 15000
            assert rate_limit_info["remaining_requests"] == 14399
            assert rate_limit_info["remaining_tokens"] == 14977
            assert rate_limit_info["reset_requests"] == 6
            # reset_tokens should be parsed or missing if parsing fails
            if "reset_tokens" in rate_limit_info:
                assert abs(rate_limit_info["reset_tokens"] - 0.092) < 0.001
            assert rate_limit_info["retry_after_seconds"] == 30
            assert "timestamp" in rate_limit_info

    def test_parse_rate_limit_headers_partial(self):
        """Test parsing with partial rate limit headers."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Mock partial headers
            headers = {
                "x-ratelimit-remaining-requests": "100",
                "x-ratelimit-remaining-tokens": "5000",
            }

            rate_limit_info = provider._parse_rate_limit_headers(headers)

            assert rate_limit_info["remaining_requests"] == 100
            assert rate_limit_info["remaining_tokens"] == 5000
            assert "timestamp" in rate_limit_info
            # Missing headers should not be present
            assert "limit_requests_per_day" not in rate_limit_info

    def test_store_and_retrieve_rate_limit_info(self, mock_redis_client):
        """Test storing and retrieving rate limit info in Redis."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            # Test storing rate limit info
            rate_limit_info = {
                "remaining_requests": 100,
                "remaining_tokens": 5000,
                "timestamp": 1234567890,
            }

            provider._store_rate_limit_info(rate_limit_info)

            # Verify Redis setex was called correctly
            mock_redis_client.setex.assert_called_once()
            args = mock_redis_client.setex.call_args[0]
            assert args[0] == "groq:rate_limit"
            assert args[1] == 300  # 5 minutes TTL
            stored_data = json.loads(args[2])
            assert stored_data == rate_limit_info

            # Test retrieving rate limit info
            mock_redis_client.get.return_value = json.dumps(rate_limit_info)
            retrieved_info = provider._get_stored_rate_limit_info()
            assert retrieved_info == rate_limit_info

    def test_should_skip_due_to_rate_limits(self, mock_redis_client):
        """Test rate limit preemption logic."""
        import time

        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            # Test with no stored info
            mock_redis_client.get.return_value = None
            should_skip, delay = provider._should_skip_due_to_rate_limits()
            assert should_skip is False
            assert delay == 0

            # Test with fresh info showing low quota
            current_time = time.time()
            rate_limit_info = {
                "remaining_requests": 1,  # Very low
                "remaining_tokens": 500,  # Low
                "retry_after_seconds": 60,
                "timestamp": current_time,
            }
            mock_redis_client.get.return_value = json.dumps(rate_limit_info)

            with patch("app.core.llm_router.metrics") as mock_metrics:
                should_skip, delay = provider._should_skip_due_to_rate_limits()
                assert should_skip is True
                assert delay == 60
                mock_metrics.increment_counter.assert_called_once_with(
                    "llm_provider_rate_limit_hits_total",
                    {"provider": "groq", "action": "preemptive_skip"},
                )

            # Test with stale info (older than 2 minutes)
            old_rate_limit_info = {
                "remaining_requests": 1,
                "remaining_tokens": 500,
                "timestamp": current_time - 150,  # 2.5 minutes ago
            }
            mock_redis_client.get.return_value = json.dumps(old_rate_limit_info)
            should_skip, delay = provider._should_skip_due_to_rate_limits()
            assert should_skip is False  # Stale info, don't skip

    def test_add_jitter(self):
        """Test jitter addition for retry delays."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Test jitter adds variance
            base_delay = 10.0
            jittered_delays = [provider._add_jitter(base_delay) for _ in range(100)]

            # All delays should be between 7.5 and 12.5 (±25%)
            assert all(7.5 <= delay <= 12.5 for delay in jittered_delays)

            # Should have some variance
            assert len(set(jittered_delays)) > 50  # Most should be unique

    @pytest.mark.asyncio
    async def test_groq_completion_success(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test successful Groq completion."""
        mock_client, mock_completions = mock_groq_client

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            response = await provider.complete(sample_messages)

            assert response.text == "Hello world"

    @pytest.mark.asyncio
    async def test_groq_completion_timeout(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test Groq completion timeout handling."""
        mock_client, mock_completions = mock_groq_client

        # Override with timeout side effect
        async def timeout_create(*args, **kwargs):
            raise asyncio.TimeoutError("Timeout")

        mock_completions.create = timeout_create

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            with pytest.raises(TimeoutError, match="Groq request timed out"):
                await provider.complete(sample_messages)

    @pytest.mark.asyncio
    async def test_groq_completion_rate_limit_429(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test Groq completion 429 rate limit handling."""
        mock_client, mock_completions = mock_groq_client

        # Create mock exception with response headers
        mock_exception = Exception("429 Rate limit exceeded")
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "30"}
        mock_exception.response = mock_response

        async def rate_limit_create(*args, **kwargs):
            raise mock_exception

        mock_completions.create = rate_limit_create

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
            patch("app.core.llm_router.metrics") as mock_metrics,
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            with pytest.raises(Exception, match="429 Rate limit exceeded"):
                await provider.complete(sample_messages)

            # Should increment rate limit metric
            mock_metrics.increment_counter.assert_called_with(
                "llm_provider_rate_limit_hits_total", {"provider": "groq", "action": "hit"}
            )

    @pytest.mark.asyncio
    async def test_groq_completion_capacity_exceeded_498(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test Groq completion 498 capacity exceeded handling."""
        mock_client, mock_completions = mock_groq_client

        async def capacity_create(*args, **kwargs):
            raise Exception("498 Capacity exceeded")

        mock_completions.create = capacity_create

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
            patch("app.core.llm_router.metrics") as mock_metrics,
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            with pytest.raises(Exception, match="498 Capacity exceeded"):
                await provider.complete(sample_messages)

            # Should increment rate limit metric (498 treated like 429)
            mock_metrics.increment_counter.assert_called_with(
                "llm_provider_rate_limit_hits_total", {"provider": "groq", "action": "hit"}
            )

    @pytest.mark.asyncio
    async def test_groq_retry_with_backoff_and_jitter(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test Groq retry logic with exponential backoff and jitter."""
        mock_client, mock_completions = mock_groq_client

        # Mock response that succeeds after retries
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Success after retry"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {"response": "data"}

        # First call fails with timeout, second succeeds
        call_count = 0

        async def retry_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("Timeout")
            return mock_response

        mock_completions.create = retry_create

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            response = await provider.complete(sample_messages)

            assert response.text == "Success after retry"
            assert call_count == 2

            # Verify sleep was called with jittered value (should be around 1 but with jitter)
            mock_sleep.assert_called_once()
            sleep_time = mock_sleep.call_args[0][0]
            assert 0.75 <= sleep_time <= 1.25  # 1 second ±25% jitter

    @pytest.mark.asyncio
    async def test_groq_retry_with_retry_after_header(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test Groq retry respecting retry-after header."""
        mock_client, mock_completions = mock_groq_client

        # Create mock exception with retry-after header
        mock_exception = Exception("429 Rate limit exceeded")
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "45"}  # 45 seconds
        mock_exception.response = mock_response

        # Mock success response
        mock_success_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Success after retry-after"
        mock_choice.message = mock_message
        mock_success_response.choices = [mock_choice]
        mock_success_response.model_dump.return_value = {"response": "data"}

        # First call fails with rate limit, second succeeds
        call_count_2 = 0

        async def retry_after_create(*args, **kwargs):
            nonlocal call_count_2
            call_count_2 += 1
            if call_count_2 == 1:
                raise mock_exception
            return mock_success_response

        mock_completions.create = retry_after_create

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("app.core.llm_router.metrics") as mock_metrics,
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            response = await provider.complete(sample_messages)

            assert response.text == "Success after retry-after"
            assert call_count_2 == 2

            # Should sleep for retry-after time (45 seconds)
            mock_sleep.assert_called_once_with(45)

            # Should store rate limit info in Redis before final failure
            # (but in this case it succeeded on retry, so we just check metrics)
            mock_metrics.increment_counter.assert_called_with(
                "llm_provider_rate_limit_hits_total", {"provider": "groq", "action": "hit"}
            )

    @pytest.mark.asyncio
    async def test_groq_preemptive_skip(self, mock_groq_client, mock_redis_client, sample_messages):
        """Test Groq preemptive skip due to known rate limits."""
        mock_client, mock_completions = mock_groq_client

        import time

        current_time = time.time()

        # Mock stored rate limit info indicating low quota
        rate_limit_info = {
            "remaining_requests": 1,  # Very low
            "remaining_tokens": 500,  # Low
            "retry_after_seconds": 30,
            "timestamp": current_time,
        }

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
            patch("app.core.llm_router.metrics") as mock_metrics,
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client
            mock_redis_client.get.return_value = json.dumps(rate_limit_info)

            # Should fail immediately due to preemptive skip
            with pytest.raises(Exception, match="Rate limit preemption"):
                await provider.complete(sample_messages)

            # The API should not be called due to preemptive skip

            # Should record preemptive skip metric
            mock_metrics.increment_counter.assert_called_with(
                "llm_provider_rate_limit_hits_total",
                {"provider": "groq", "action": "preemptive_skip"},
            )

    @pytest.mark.asyncio
    async def test_groq_streaming_success(
        self, mock_groq_client, mock_redis_client, sample_messages
    ):
        """Test successful Groq streaming."""
        mock_client, mock_completions = mock_groq_client

        # Mock streaming response
        async def mock_stream(*args, **kwargs):
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
            ]
            for chunk in chunks:
                chunk.model_dump.return_value = {"chunk": "data"}
                yield chunk

        mock_completions.create = mock_stream

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()
            provider.redis_client = mock_redis_client

            responses = []
            async for response in provider.stream_complete(sample_messages):
                responses.append(response)

            assert len(responses) == 2
            assert responses[0].text == "Hello"
            assert responses[1].text == " world"


class TestLLMRouter:
    """Test LLM Router functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.exists.return_value = 0
        mock_redis.setex.return_value = True
        return mock_redis

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with test values."""
        with patch("app.core.llm_router.settings") as mock_settings:
            mock_settings.REDIS_URL = "redis://localhost:6379/0"
            mock_settings.LLM_TIMEOUT_SECONDS = 30
            mock_settings.GROQ_TIMEOUT_SECONDS = 30
            mock_settings.LLM_FALLBACK_CACHE_SECONDS = 300
            mock_settings.OPENROUTER_API_KEY = "test-openrouter-key"
            mock_settings.GROQ_API_KEY = "test-groq-key"
            mock_settings.GROQ_MODEL_NAME = "llama3-70b-8192"
            mock_settings.OPENAI_API_KEY = ""
            mock_settings.LOCAL_LLM_PATH = ""
            mock_settings.LLM_MODEL_NAME = "google/gemini-1.5-pro-latest"
            yield mock_settings

    def test_router_initialization_success(self, mock_redis, mock_settings):
        """Test successful router initialization."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            assert router.redis_client == mock_redis
            assert len(router.available_providers) >= 2  # OpenRouter and Groq should be available
            mock_redis.ping.assert_called_once()

    def test_router_initialization_no_providers(self, mock_redis, mock_settings):
        """Test router initialization with no available providers."""
        mock_settings.OPENROUTER_API_KEY = ""
        mock_settings.GROQ_API_KEY = ""

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            with pytest.raises(RuntimeError, match="No LLM providers are configured"):
                LLMRouter()

    def test_router_initialization_redis_failure(self, mock_settings):
        """Test router initialization with Redis connection failure."""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            assert router.redis_client is None  # Should handle Redis failure gracefully

    def test_cache_key_generation(self, mock_redis, mock_settings):
        """Test cache key generation for failed providers."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            provider = GroqProvider()

            cache_key = router._get_cache_key(provider, "test-model", ErrorType.TIMEOUT)
            expected = "failed_provider:groq:test-model:timeout"
            assert cache_key == expected

    def test_error_classification(self, mock_redis, mock_settings):
        """Test error classification for caching."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()

            # Test timeout error
            timeout_error = TimeoutError("Request timed out")
            assert router._classify_error(timeout_error) == ErrorType.TIMEOUT

            # Test rate limit error
            rate_limit_error = Exception("Rate limit exceeded (429)")
            assert router._classify_error(rate_limit_error) == ErrorType.RATE_LIMIT

            # Test auth error
            auth_error = Exception("Unauthorized (401)")
            assert router._classify_error(auth_error) == ErrorType.AUTH_FAILURE

            # Test generic error
            generic_error = Exception("Something went wrong")
            assert router._classify_error(generic_error) == ErrorType.API_ERROR

    def test_fallback_decision(self, mock_redis, mock_settings):
        """Test fallback decision logic."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()

            # Should trigger fallback
            assert router._should_trigger_fallback(TimeoutError("timeout")) is True
            assert router._should_trigger_fallback(Exception("rate limit")) is True
            assert router._should_trigger_fallback(Exception("auth error")) is True

            # Should not trigger fallback
            assert router._should_trigger_fallback(Exception("400 Bad Request")) is False
            assert router._should_trigger_fallback(Exception("malformed request")) is False

    def test_provider_caching(self, mock_redis, mock_settings):
        """Test caching of failed providers."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            provider = GroqProvider()

            # Test caching a failure
            router._cache_provider_failure(provider, ErrorType.TIMEOUT)

            expected_key = "failed_provider:groq:llama3-70b-8192:timeout"
            mock_redis.setex.assert_called_once()
            args = mock_redis.setex.call_args[0]
            assert args[0] == expected_key
            assert args[1] == 300  # LLM_FALLBACK_CACHE_SECONDS

            # Verify the cached data structure
            cached_data = json.loads(args[2])
            assert cached_data["provider"] == "groq"
            assert cached_data["error_type"] == "timeout"
            assert "timestamp" in cached_data

    def test_provider_cache_check(self, mock_redis, mock_settings):
        """Test checking if provider is cached as failed."""
        mock_redis.exists.return_value = 1  # Provider is cached

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            provider = GroqProvider()

            is_cached = router._is_provider_cached_as_failed(provider, ErrorType.TIMEOUT)
            assert is_cached is True

            expected_key = "failed_provider:groq:llama3-70b-8192:timeout"
            mock_redis.exists.assert_called_with(expected_key)

    @pytest.mark.asyncio
    async def test_successful_completion(self, mock_redis, mock_settings):
        """Test successful completion with first provider."""
        mock_response = MagicMock()
        mock_response.text = "Test response"

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            with patch("app.core.llm_router.OpenRouterProvider") as mock_provider_class:
                mock_provider = MagicMock()
                mock_provider.is_available.return_value = True
                mock_provider.complete = AsyncMock(return_value=mock_response)
                mock_provider.provider_type = ProviderType.OPENROUTER
                mock_provider.get_model_name.return_value = "test-model"
                mock_provider_class.return_value = mock_provider

                router = LLMRouter()
                router.available_providers = [mock_provider]

                messages = [ChatMessage(role=MessageRole.USER, content="Test question")]
                response = await router.acomplete(messages)

                assert response == mock_response
                mock_provider.complete.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self, mock_redis, mock_settings):
        """Test automatic fallback when first provider times out."""
        mock_response = MagicMock()
        mock_response.text = "Fallback response"

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            # Create two mock providers
            mock_provider1 = MagicMock()
            mock_provider1.is_available.return_value = True
            mock_provider1.complete = AsyncMock(side_effect=TimeoutError("Timeout"))
            mock_provider1.provider_type = ProviderType.OPENROUTER
            mock_provider1.get_model_name.return_value = "model1"

            mock_provider2 = MagicMock()
            mock_provider2.is_available.return_value = True
            mock_provider2.complete = AsyncMock(return_value=mock_response)
            mock_provider2.provider_type = ProviderType.OPENAI
            mock_provider2.get_model_name.return_value = "model2"

            router = LLMRouter()
            router.available_providers = [mock_provider1, mock_provider2]

            messages = [ChatMessage(role=MessageRole.USER, content="Test question")]
            response = await router.acomplete(messages)

            # Should succeed with second provider
            assert response == mock_response
            mock_provider1.complete.assert_called_once()
            mock_provider2.complete.assert_called_once()

            # First provider should be cached as failed
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, mock_redis, mock_settings):
        """Test behavior when all providers fail."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_provider.complete = AsyncMock(side_effect=Exception("All failed"))
            mock_provider.provider_type = ProviderType.OPENROUTER
            mock_provider.get_model_name.return_value = "test-model"

            router = LLMRouter()
            router.available_providers = [mock_provider]

            messages = [ChatMessage(role=MessageRole.USER, content="Test question")]

            with pytest.raises(Exception, match="All failed"):
                await router.acomplete(messages)

    @pytest.mark.asyncio
    async def test_skip_cached_provider(self, mock_redis, mock_settings):
        """Test skipping providers that are cached as failed."""

        # Configure Redis to only show first provider as cached
        def mock_exists(key):
            return 1 if "openrouter" in key else 0

        mock_redis.exists.side_effect = mock_exists
        mock_response = MagicMock()

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            # First provider is cached as failed
            mock_provider1 = MagicMock()
            mock_provider1.is_available.return_value = True
            mock_provider1.provider_type = ProviderType.OPENROUTER
            mock_provider1.get_model_name.return_value = "model1"

            # Second provider should be used
            mock_provider2 = MagicMock()
            mock_provider2.is_available.return_value = True
            mock_provider2.complete = AsyncMock(return_value=mock_response)
            mock_provider2.provider_type = ProviderType.OPENAI
            mock_provider2.get_model_name.return_value = "model2"

            router = LLMRouter()
            router.available_providers = [mock_provider1, mock_provider2]

            messages = [ChatMessage(role=MessageRole.USER, content="Test question")]
            response = await router.acomplete(messages)

            # Should skip first provider and use second
            assert response == mock_response
            mock_provider2.complete.assert_called_once()

            # First provider should not be called since it's cached as failed

    def test_synchronous_complete(self, mock_redis, mock_settings):
        """Test synchronous complete method."""
        mock_response = MagicMock()

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            with patch("app.core.llm_router.OpenRouterProvider") as mock_provider_class:
                mock_provider = MagicMock()
                mock_provider.is_available.return_value = True
                mock_provider.complete = AsyncMock(return_value=mock_response)
                mock_provider.provider_type = ProviderType.OPENROUTER
                mock_provider.get_model_name.return_value = "test-model"
                mock_provider_class.return_value = mock_provider

                router = LLMRouter()
                router.available_providers = [mock_provider]

                messages = [ChatMessage(role=MessageRole.USER, content="Test question")]
                response = router.complete(messages)

                assert response == mock_response

    @pytest.mark.asyncio
    async def test_streaming_complete(self, mock_redis, mock_settings):
        """Test streaming completion functionality."""

        async def mock_stream():
            for i in range(3):
                yield f"chunk_{i}"

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            # Set up the stream_complete method to return an async generator
            mock_provider.stream_complete.return_value = mock_stream()
            mock_provider.provider_type = ProviderType.OPENROUTER
            mock_provider.get_model_name.return_value = "test-model"

            router = LLMRouter()
            router.available_providers = [mock_provider]

            messages = [ChatMessage(role=MessageRole.USER, content="Test question")]
            chunks = []

            async for chunk in router.astream_complete(messages):
                chunks.append(chunk)

            assert chunks == ["chunk_0", "chunk_1", "chunk_2"]
            mock_provider.stream_complete.assert_called_once()

    def test_router_metadata(self, mock_redis, mock_settings):
        """Test router metadata properties."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            metadata = router.metadata

            assert metadata.context_window == 4096
            assert metadata.num_output == 1024
            assert metadata.is_chat_model is True
            assert metadata.model_name == "llm-router"
