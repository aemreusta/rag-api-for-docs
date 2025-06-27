"""
Unit tests for the LLM Router implementation.

Tests cover:
- Provider availability detection
- Automatic fallback on errors (timeout, rate limit, auth failures)
- Redis caching of failed providers
- Error classification and handling
- Streaming and non-streaming responses
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

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

    def test_groq_provider_placeholder(self):
        """Test Groq provider placeholder implementation."""
        provider = GroqProvider()
        assert provider.is_available() is False
        assert provider.provider_type == ProviderType.GROQ
        assert provider.get_model_name() == "llama3-70b-8192"

    def test_local_provider_placeholder(self):
        """Test Local provider placeholder implementation."""
        provider = LocalProvider()
        assert provider.is_available() is False
        assert provider.provider_type == ProviderType.LOCAL
        assert provider.get_model_name() == "local-llm"


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
            mock_settings.LLM_FALLBACK_CACHE_SECONDS = 300
            mock_settings.OPENROUTER_API_KEY = "test-openrouter-key"
            mock_settings.GROQ_API_KEY = ""
            mock_settings.OPENAI_API_KEY = ""
            mock_settings.LOCAL_LLM_PATH = ""
            mock_settings.LLM_MODEL_NAME = "google/gemini-1.5-pro-latest"
            yield mock_settings

    def test_router_initialization_success(self, mock_redis, mock_settings):
        """Test successful router initialization."""
        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            assert router.redis_client == mock_redis
            assert len(router.available_providers) >= 1  # At least OpenRouter should be available
            mock_redis.ping.assert_called_once()

    def test_router_initialization_no_providers(self, mock_redis, mock_settings):
        """Test router initialization with no available providers."""
        mock_settings.OPENROUTER_API_KEY = ""

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
            provider = OpenRouterProvider()

            cache_key = router._get_cache_key(provider, "test-model", ErrorType.TIMEOUT)
            expected = "failed_provider:openrouter:test-model:timeout"
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
            provider = OpenRouterProvider()

            # Test caching a failure
            router._cache_provider_failure(provider, ErrorType.TIMEOUT)

            expected_key = "failed_provider:openrouter:google/gemini-1.5-pro-latest:timeout"
            mock_redis.setex.assert_called_once()
            args = mock_redis.setex.call_args[0]
            assert args[0] == expected_key
            assert args[1] == 300  # LLM_FALLBACK_CACHE_SECONDS

            # Verify the cached data structure
            cached_data = json.loads(args[2])
            assert cached_data["provider"] == "openrouter"
            assert cached_data["error_type"] == "timeout"
            assert "timestamp" in cached_data

    def test_provider_cache_check(self, mock_redis, mock_settings):
        """Test checking if provider is cached as failed."""
        mock_redis.exists.return_value = 1  # Provider is cached

        with patch("app.core.llm_router.redis.from_url", return_value=mock_redis):
            router = LLMRouter()
            provider = OpenRouterProvider()

            is_cached = router._is_provider_cached_as_failed(provider, ErrorType.TIMEOUT)
            assert is_cached is True

            expected_key = "failed_provider:openrouter:google/gemini-1.5-pro-latest:timeout"
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
