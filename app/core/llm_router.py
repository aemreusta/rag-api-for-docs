"""
LLM Router implementation with automatic fallback and provider management.

This module provides a unified interface for multiple LLM providers with:
- Priority-based routing: Gemini(OpenRouter) → Groq(Llama3) → ChatGPT → Local
- Automatic fallback on errors (timeout, rate limit, auth failures)
- Redis-based caching of failed providers (5-minute cooldown)
- Configurable timeouts per provider
- Support for streaming and non-streaming responses
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

import redis
from llama_index.core.base.llms.types import ChatMessage, CompletionResponse
from llama_index.core.llms import LLMMetadata
from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from pydantic import PrivateAttr

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ProviderType(Enum):
    """Enumeration of supported LLM providers."""

    OPENROUTER = "openrouter"
    GROQ = "groq"
    OPENAI = "openai"
    LOCAL = "local"


class ErrorType(Enum):
    """Enumeration of error types for caching failed providers."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_FAILURE = "auth_failure"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, provider_type: ProviderType, timeout_seconds: int = 30):
        self.provider_type = provider_type
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(f"{__name__}.{provider_type.value}")

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """Generate completion for the given messages."""
        pass

    @abstractmethod
    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Generate streaming completion for the given messages."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name for this provider."""
        pass


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation for Gemini models."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(ProviderType.OPENROUTER, timeout_seconds)
        if self.is_available():
            self.client = OpenRouter(
                api_key=settings.OPENROUTER_API_KEY,
                model=settings.LLM_MODEL_NAME,
                timeout=timeout_seconds,
            )

    def is_available(self) -> bool:
        return bool(settings.OPENROUTER_API_KEY)

    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        try:
            self.logger.info("Calling OpenRouter for completion", model=settings.LLM_MODEL_NAME)
            response = await asyncio.wait_for(
                self.client.acomplete(messages, **kwargs), timeout=self.timeout_seconds
            )
            self.logger.info("OpenRouter completion successful")
            return response
        except asyncio.TimeoutError as err:
            self.logger.warning("OpenRouter request timed out", timeout=self.timeout_seconds)
            raise TimeoutError(
                f"OpenRouter request timed out after {self.timeout_seconds}s"
            ) from err
        except Exception as e:
            self.logger.error(
                "OpenRouter completion failed", error=str(e), error_type=type(e).__name__
            )
            raise

    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        try:
            self.logger.info(
                "Calling OpenRouter for streaming completion", model=settings.LLM_MODEL_NAME
            )
            async for chunk in self.client.astream_complete(messages, **kwargs):
                yield chunk
            self.logger.info("OpenRouter streaming completion successful")
        except asyncio.TimeoutError as err:
            self.logger.warning(
                "OpenRouter streaming request timed out", timeout=self.timeout_seconds
            )
            raise TimeoutError(
                f"OpenRouter streaming request timed out after {self.timeout_seconds}s"
            ) from err
        except Exception as e:
            self.logger.error(
                "OpenRouter streaming completion failed", error=str(e), error_type=type(e).__name__
            )
            raise

    def get_model_name(self) -> str:
        return settings.LLM_MODEL_NAME


class GroqProvider(LLMProvider):
    """Groq provider implementation (placeholder for future integration)."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(ProviderType.GROQ, timeout_seconds)
        # TODO: Initialize Groq client when feat/groq-llama3 is implemented
        self.model_name = "llama3-70b-8192"

    def is_available(self) -> bool:
        # TODO: Check GROQ_API_KEY when implementation is ready
        return False  # Placeholder - not implemented yet

    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        # TODO: Implement Groq completion
        raise NotImplementedError(
            "Groq provider not yet implemented - placeholder for feat/groq-llama3"
        )

    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        # TODO: Implement Groq streaming
        raise NotImplementedError(
            "Groq provider not yet implemented - placeholder for feat/groq-llama3"
        )
        yield  # This yield is needed to make this a generator function

    def get_model_name(self) -> str:
        return self.model_name


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation for ChatGPT models."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(ProviderType.OPENAI, timeout_seconds)
        self.model_name = "gpt-3.5-turbo"
        if self.is_available():
            self.client = OpenAI(
                api_key=settings.OPENAI_API_KEY, model=self.model_name, timeout=timeout_seconds
            )

    def is_available(self) -> bool:
        return bool(settings.OPENAI_API_KEY)

    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        try:
            self.logger.info("Calling OpenAI for completion", model=self.model_name)
            response = await asyncio.wait_for(
                self.client.acomplete(messages, **kwargs), timeout=self.timeout_seconds
            )
            self.logger.info("OpenAI completion successful")
            return response
        except asyncio.TimeoutError as err:
            self.logger.warning("OpenAI request timed out", timeout=self.timeout_seconds)
            raise TimeoutError(f"OpenAI request timed out after {self.timeout_seconds}s") from err
        except Exception as e:
            self.logger.error("OpenAI completion failed", error=str(e), error_type=type(e).__name__)
            raise

    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        try:
            self.logger.info("Calling OpenAI for streaming completion", model=self.model_name)
            async for chunk in self.client.astream_complete(messages, **kwargs):
                yield chunk
            self.logger.info("OpenAI streaming completion successful")
        except asyncio.TimeoutError as err:
            self.logger.warning("OpenAI streaming request timed out", timeout=self.timeout_seconds)
            raise TimeoutError(
                f"OpenAI streaming request timed out after {self.timeout_seconds}s"
            ) from err
        except Exception as e:
            self.logger.error(
                "OpenAI streaming completion failed", error=str(e), error_type=type(e).__name__
            )
            raise

    def get_model_name(self) -> str:
        return self.model_name


class LocalProvider(LLMProvider):
    """Local LLM provider implementation (placeholder for future implementation)."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(ProviderType.LOCAL, timeout_seconds)
        self.model_name = "local-llm"

    def is_available(self) -> bool:
        # TODO: Check if local model is available when implemented
        return False  # Placeholder - not implemented yet

    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        # TODO: Implement local LLM completion
        raise NotImplementedError("Local LLM provider not yet implemented")

    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        # TODO: Implement local LLM streaming
        raise NotImplementedError("Local LLM provider not yet implemented")
        yield  # This yield is needed to make this a generator function

    def get_model_name(self) -> str:
        return self.model_name


class LLMRouter(CustomLLM):
    """
    LLM Router that manages multiple providers with automatic fallback.

    Implements priority-based routing: Gemini(OpenRouter) → Groq → ChatGPT → Local
    Caches failed providers in Redis for 5-minute cooldown periods.
    """

    # Private attributes for Pydantic
    _redis_client: Any | None = PrivateAttr(default=None)
    _providers: list[LLMProvider] = PrivateAttr(default_factory=list)
    _available_providers: list[LLMProvider] = PrivateAttr(default_factory=list)

    def __init__(self):
        super().__init__()

        # Initialize Redis client for caching failed providers
        try:
            self._redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self._redis_client.ping()  # Test connection
            logger.info("Redis connection established for LLM Router")
        except Exception as e:
            logger.error("Failed to connect to Redis for LLM Router", error=str(e))
            self._redis_client = None

        # Initialize providers in priority order
        self._providers = [
            OpenRouterProvider(settings.LLM_TIMEOUT_SECONDS),
            GroqProvider(settings.LLM_TIMEOUT_SECONDS),
            OpenAIProvider(settings.LLM_TIMEOUT_SECONDS),
            LocalProvider(settings.LLM_TIMEOUT_SECONDS),
        ]

        # Filter to only available providers
        self._available_providers = [p for p in self._providers if p.is_available()]

        if not self._available_providers:
            logger.error("No LLM providers are available!")
            raise RuntimeError("No LLM providers are configured and available")

        logger.info(
            "LLM Router initialized",
            available_providers=[p.provider_type.value for p in self._available_providers],
            total_providers=len(self._providers),
        )

    @property
    def redis_client(self):
        """Access to Redis client."""
        return self._redis_client

    @redis_client.setter
    def redis_client(self, value):
        """Set Redis client (for testing)."""
        self._redis_client = value

    @property
    def available_providers(self):
        """Access to available providers."""
        return self._available_providers

    @available_providers.setter
    def available_providers(self, value):
        """Set available providers (for testing)."""
        self._available_providers = value

    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata for the router."""
        return LLMMetadata(
            context_window=4096,  # Conservative estimate
            num_output=1024,
            is_chat_model=True,
            model_name="llm-router",
        )

    def _get_cache_key(self, provider: LLMProvider, model: str, error_type: ErrorType) -> str:
        """Generate Redis cache key for failed provider."""
        return f"failed_provider:{provider.provider_type.value}:{model}:{error_type.value}"

    def _is_provider_cached_as_failed(self, provider: LLMProvider, error_type: ErrorType) -> bool:
        """Check if provider is cached as failed for the given error type."""
        if not self._redis_client:
            return False

        cache_key = self._get_cache_key(provider, provider.get_model_name(), error_type)
        try:
            return self._redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.warning("Failed to check provider cache", error=str(e))
            return False

    def _cache_provider_failure(self, provider: LLMProvider, error_type: ErrorType):
        """Cache provider as failed for the specified duration."""
        if not self._redis_client:
            return

        cache_key = self._get_cache_key(provider, provider.get_model_name(), error_type)
        try:
            self._redis_client.setex(
                cache_key,
                settings.LLM_FALLBACK_CACHE_SECONDS,
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "provider": provider.provider_type.value,
                        "model": provider.get_model_name(),
                        "error_type": error_type.value,
                    }
                ),
            )
            logger.info(
                "Cached provider failure",
                provider=provider.provider_type.value,
                model=provider.get_model_name(),
                error_type=error_type.value,
                cache_duration=settings.LLM_FALLBACK_CACHE_SECONDS,
            )
        except Exception as e:
            logger.warning("Failed to cache provider failure", error=str(e))

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for caching purposes."""
        error_str = str(error).lower()
        type(error).__name__.lower()

        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "unauthorized" in error_str or "401" in error_str or "403" in error_str:
            return ErrorType.AUTH_FAILURE
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.API_ERROR

    def _should_trigger_fallback(self, error: Exception) -> bool:
        """Determine if error should trigger fallback to next provider."""
        # Don't fallback for malformed requests or client errors
        error_str = str(error).lower()
        if "400" in error_str or "malformed" in error_str or "invalid request" in error_str:
            return False

        # Fallback for all other errors (timeouts, rate limits, auth failures, etc.)
        return True

    async def _try_provider(
        self, provider: LLMProvider, messages: list[ChatMessage], **kwargs
    ) -> CompletionResponse:
        """Try to get completion from a specific provider."""
        try:
            logger.info(
                "Attempting provider",
                provider=provider.provider_type.value,
                model=provider.get_model_name(),
            )

            response = await provider.complete(messages, **kwargs)

            logger.info(
                "Provider succeeded",
                provider=provider.provider_type.value,
                model=provider.get_model_name(),
            )

            return response

        except Exception as e:
            error_type = self._classify_error(e)

            logger.warning(
                "Provider failed",
                provider=provider.provider_type.value,
                model=provider.get_model_name(),
                error=str(e),
                error_type=error_type.value,
            )

            # Cache the failure
            self._cache_provider_failure(provider, error_type)

            if not self._should_trigger_fallback(e):
                # Re-raise immediately for client errors
                raise

            # Otherwise, let the router try the next provider
            raise

    async def acomplete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """Complete method with automatic fallback across providers."""
        last_error = None

        for provider in self._available_providers:
            # Skip providers that are cached as failed
            error_type = ErrorType.API_ERROR  # Check general failure first
            if self._is_provider_cached_as_failed(provider, error_type):
                logger.info(
                    "Skipping cached failed provider",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                continue

            try:
                return await self._try_provider(provider, messages, **kwargs)
            except Exception as e:
                last_error = e
                continue

        # All providers failed
        logger.error(
            "All LLM providers failed",
            attempted_providers=[p.provider_type.value for p in self._available_providers],
            last_error=str(last_error) if last_error else "Unknown error",
        )

        if last_error:
            raise last_error
        else:
            raise RuntimeError("All LLM providers failed")

    def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """Synchronous complete method."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.acomplete(messages, **kwargs))

    async def astream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Streaming complete method with automatic fallback across providers."""
        last_error = None

        for provider in self._available_providers:
            # Skip providers that are cached as failed
            error_type = ErrorType.API_ERROR
            if self._is_provider_cached_as_failed(provider, error_type):
                logger.info(
                    "Skipping cached failed provider for streaming",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                continue

            try:
                logger.info(
                    "Attempting streaming with provider",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )

                async for chunk in provider.stream_complete(messages, **kwargs):
                    yield chunk

                logger.info(
                    "Streaming completed successfully",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                return

            except Exception as e:
                error_type = self._classify_error(e)

                logger.warning(
                    "Provider streaming failed",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                    error=str(e),
                    error_type=error_type.value,
                )

                # Cache the failure
                self._cache_provider_failure(provider, error_type)

                if not self._should_trigger_fallback(e):
                    # Re-raise immediately for client errors
                    raise

                last_error = e
                continue

        # All providers failed
        logger.error(
            "All LLM providers failed for streaming",
            attempted_providers=[p.provider_type.value for p in self._available_providers],
            last_error=str(last_error) if last_error else "Unknown error",
        )

        if last_error:
            raise last_error
        else:
            raise RuntimeError("All LLM providers failed for streaming")

    def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Synchronous streaming complete method."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _stream():
            async for chunk in self.astream_complete(messages, **kwargs):
                yield chunk

        # Convert async generator to sync generator
        async_gen = _stream()
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break
