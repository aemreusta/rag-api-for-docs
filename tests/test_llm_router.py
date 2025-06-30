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
        mock_client = AsyncMock()
        mock_chat = AsyncMock()
        mock_completions = AsyncMock()
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        return mock_client, mock_completions

    @pytest.fixture
    def sample_messages(self):
        """Create sample chat messages for testing."""
        return [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="Tell me about LLMs."),
        ]

    def test_convert_messages_to_groq_format(self):
        """Test conversion of LlamaIndex messages to Groq format."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are helpful"),
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ]

            groq_messages = provider._convert_messages_to_groq_format(messages)

            expected = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
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
            mock_message.content = "This is a test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.model_dump.return_value = {"test": "data"}

            response = provider._create_completion_response(mock_response)

            assert response.text == "This is a test response"
            assert response.raw == {"test": "data"}

    def test_create_streaming_response(self):
        """Test conversion of Groq streaming chunk to LlamaIndex format."""
        with patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"):
            provider = GroqProvider()

            # Mock Groq streaming chunk
            mock_chunk = MagicMock()
            mock_choice = MagicMock()
            mock_delta = MagicMock()
            mock_delta.content = "streaming"
            mock_choice.delta = mock_delta
            mock_chunk.choices = [mock_choice]
            mock_chunk.model_dump.return_value = {"chunk": "data"}

            response = provider._create_streaming_response(mock_chunk)

            assert response.text == "streaming"
            assert response.raw == {"chunk": "data"}

    @pytest.mark.asyncio
    async def test_groq_completion_success(self, mock_groq_client, sample_messages):
        """Test successful Groq completion."""
        mock_client, mock_completions = mock_groq_client

        # Mock successful response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "LLMs are large language models."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.model_dump.return_value = {"response": "data"}

        mock_completions.create.return_value = mock_response

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()
            response = await provider.complete(sample_messages)

            assert response.text == "LLMs are large language models."

            # Verify API call
            mock_completions.create.assert_called_once()
            call_args = mock_completions.create.call_args[1]
            assert call_args["model"] == "llama3-70b-8192"
            assert call_args["messages"] == [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about LLMs."},
            ]

    @pytest.mark.asyncio
    async def test_groq_completion_timeout(self, mock_groq_client, sample_messages):
        """Test Groq completion timeout handling."""
        mock_client, mock_completions = mock_groq_client
        mock_completions.create.side_effect = asyncio.TimeoutError("Request timed out")

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()

            with pytest.raises(TimeoutError, match="Groq request timed out"):
                await provider.complete(sample_messages)

    @pytest.mark.asyncio
    async def test_groq_completion_rate_limit(self, mock_groq_client, sample_messages):
        """Test Groq completion rate limit handling."""
        mock_client, mock_completions = mock_groq_client
        mock_completions.create.side_effect = Exception("429 Rate limit exceeded")

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()

            with pytest.raises(Exception, match="429 Rate limit exceeded"):
                await provider.complete(sample_messages)

    @pytest.mark.asyncio
    async def test_groq_streaming_success(self, mock_groq_client, sample_messages):
        """Test successful Groq streaming."""
        mock_client, mock_completions = mock_groq_client

        # Mock streaming response
        mock_stream = AsyncMock()
        mock_chunk1 = MagicMock()
        mock_chunk2 = MagicMock()

        mock_choice1 = MagicMock()
        mock_delta1 = MagicMock()
        mock_delta1.content = "Hello"
        mock_choice1.delta = mock_delta1
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.model_dump.return_value = {"chunk1": "data"}

        mock_choice2 = MagicMock()
        mock_delta2 = MagicMock()
        mock_delta2.content = " world"
        mock_choice2.delta = mock_delta2
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.model_dump.return_value = {"chunk2": "data"}

        mock_stream.__aiter__.return_value = [mock_chunk1, mock_chunk2]
        mock_completions.create.return_value = mock_stream

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
        ):
            provider = GroqProvider()
            responses = []

            async for response in provider.stream_complete(sample_messages):
                responses.append(response)

            assert len(responses) == 2
            assert responses[0].text == "Hello"
            assert responses[1].text == " world"

    @pytest.mark.asyncio
    async def test_groq_retry_with_backoff(self, mock_groq_client, sample_messages):
        """Test Groq retry logic with exponential backoff."""
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
        mock_completions.create.side_effect = [asyncio.TimeoutError("Timeout"), mock_response]

        with (
            patch("app.core.llm_router.settings.GROQ_API_KEY", "test-key"),
            patch("app.core.llm_router.AsyncGroq", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            provider = GroqProvider()
            response = await provider.complete(sample_messages)

            assert response.text == "Success after retry"
            assert mock_completions.create.call_count == 2
            mock_sleep.assert_called_once_with(1)  # 2^0 = 1 second


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
