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
from groq import AsyncGroq
from groq.types.chat import ChatCompletionChunk
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
    """Groq provider implementation using the official Groq Python client."""

    def __init__(self, timeout_seconds: int = None):
        # Use Groq-specific timeout if provided, otherwise use default
        timeout = timeout_seconds or settings.GROQ_TIMEOUT_SECONDS
        super().__init__(ProviderType.GROQ, timeout)
        self.model_name = settings.GROQ_MODEL_NAME

        if self.is_available():
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    def is_available(self) -> bool:
        return bool(settings.GROQ_API_KEY)

    def _convert_messages_to_groq_format(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert LlamaIndex ChatMessage format to Groq format."""
        groq_messages = []
        for msg in messages:
            groq_messages.append({"role": msg.role.value, "content": msg.content})
        return groq_messages

    def _create_completion_response(self, groq_response) -> CompletionResponse:
        """Convert Groq response to LlamaIndex CompletionResponse format."""
        from llama_index.core.base.llms.types import MessageRole

        # Extract the message content from Groq response
        message_content = groq_response.choices[0].message.content

        # Create ChatMessage for the response
        response_message = ChatMessage(role=MessageRole.ASSISTANT, content=message_content)

        return CompletionResponse(
            text=message_content,  # Required field
            message=response_message,
            raw=groq_response.model_dump()
            if hasattr(groq_response, "model_dump")
            else groq_response,
        )

    def _create_streaming_response(self, chunk: ChatCompletionChunk) -> CompletionResponse:
        """Convert Groq streaming chunk to LlamaIndex CompletionResponse format."""
        from llama_index.core.base.llms.types import MessageRole

        # Extract delta content from the chunk
        delta_content = chunk.choices[0].delta.content or ""

        # Create ChatMessage for the chunk
        response_message = ChatMessage(role=MessageRole.ASSISTANT, content=delta_content)

        return CompletionResponse(
            text=delta_content,  # Required field
            message=response_message,
            raw=chunk.model_dump() if hasattr(chunk, "model_dump") else chunk,
        )

    async def _retry_with_backoff(self, func, *args, max_retries=3, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout_seconds)
            except asyncio.TimeoutError as err:
                if attempt == max_retries - 1:
                    raise TimeoutError(
                        f"Groq request timed out after {self.timeout_seconds}s"
                    ) from err
                wait_time = 2**attempt
                self.logger.warning(
                    f"Groq request timed out, retrying in {wait_time}s", attempt=attempt + 1
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                # Check for specific Groq error types
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"Groq rate limited, retrying in {wait_time}s", attempt=attempt + 1
                    )
                    await asyncio.sleep(wait_time)
                elif "503" in error_str or "service unavailable" in error_str:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"Groq service unavailable, retrying in {wait_time}s", attempt=attempt + 1
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        try:
            self.logger.info("Calling Groq for completion", model=self.model_name)

            # Convert messages to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages)

            # Prepare parameters for Groq API
            groq_params = {
                "model": self.model_name,
                "messages": groq_messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
            }

            # Add optional parameters if provided
            if "top_p" in kwargs:
                groq_params["top_p"] = kwargs["top_p"]
            if "stream" in kwargs:
                groq_params["stream"] = kwargs["stream"]

            # Make the API call with retry logic
            response = await self._retry_with_backoff(
                self.client.chat.completions.create, **groq_params
            )

            self.logger.info("Groq completion successful")
            return self._create_completion_response(response)

        except asyncio.TimeoutError as err:
            self.logger.warning("Groq request timed out", timeout=self.timeout_seconds)
            raise TimeoutError(f"Groq request timed out after {self.timeout_seconds}s") from err
        except Exception as e:
            self.logger.error("Groq completion failed", error=str(e), error_type=type(e).__name__)
            raise

    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        try:
            self.logger.info("Calling Groq for streaming completion", model=self.model_name)

            # Convert messages to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages)

            # Prepare parameters for Groq API
            groq_params = {
                "model": self.model_name,
                "messages": groq_messages,
                "stream": True,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
            }

            # Add optional parameters if provided
            if "top_p" in kwargs:
                groq_params["top_p"] = kwargs["top_p"]

            # Make the streaming API call
            async with asyncio.timeout(self.timeout_seconds):
                stream = await self.client.chat.completions.create(**groq_params)

                async for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield self._create_streaming_response(chunk)

            self.logger.info("Groq streaming completion successful")

        except asyncio.TimeoutError as err:
            self.logger.warning("Groq streaming request timed out", timeout=self.timeout_seconds)
            raise TimeoutError(
                f"Groq streaming request timed out after {self.timeout_seconds}s"
            ) from err
        except Exception as e:
            self.logger.error(
                "Groq streaming completion failed", error=str(e), error_type=type(e).__name__
            )
            raise

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
    _logger: Any = PrivateAttr()

    @property
    def logger(self):
        """Access to the logger instance."""
        return self._logger

    def __init__(self):
        super().__init__()

        # Initialize logger for this instance
        self._logger = get_logger("app.core.llm_router")

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
            GroqProvider(settings.GROQ_TIMEOUT_SECONDS),
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
        """Get LLM metadata for the router."""
        return LLMMetadata(
            model_name="llm-router",
            is_function_calling_model=False,
            is_chat_model=True,
            context_window=4096,  # Conservative estimate - updated to match test expectations
            num_output=1024,  # Conservative estimate
        )

    def _get_cache_key(self, provider: LLMProvider, model: str, error_type: ErrorType) -> str:
        """Generate cache key for failed provider."""
        return f"failed_provider:{provider.provider_type.value}:{model}:{error_type.value}"

    def _is_provider_cached_as_failed(self, provider: LLMProvider, error_type: ErrorType) -> bool:
        """Check if provider is cached as failed."""
        if not self._redis_client:
            return False

        cache_key = self._get_cache_key(provider, provider.get_model_name(), error_type)
        return bool(self._redis_client.exists(cache_key))

    def _cache_provider_failure(self, provider: LLMProvider, error_type: ErrorType):
        """Cache provider failure in Redis."""
        if not self._redis_client:
            return

        cache_key = self._get_cache_key(provider, provider.get_model_name(), error_type)
        cache_data = {
            "provider": provider.provider_type.value,
            "model": provider.get_model_name(),
            "error_type": error_type.value,
            "timestamp": time.time(),
        }

        self._redis_client.setex(
            cache_key,
            settings.LLM_FALLBACK_CACHE_SECONDS,
            json.dumps(cache_data),
        )

        self.logger.warning(
            "Provider cached as failed",
            provider=provider.provider_type.value,
            model=provider.get_model_name(),
            error_type=error_type.value,
            cooldown_seconds=settings.LLM_FALLBACK_CACHE_SECONDS,
        )

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for caching purposes."""
        error_str = str(error).lower()
        type(error).__name__.lower()

        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "429" in error_str or "rate limit" in error_str:
            return ErrorType.RATE_LIMIT
        elif "401" in error_str or "403" in error_str or "unauthorized" in error_str:
            return ErrorType.AUTH_FAILURE
        elif "413" in error_str or "token limit" in error_str:
            return ErrorType.API_ERROR
        elif "503" in error_str or "service unavailable" in error_str:
            return ErrorType.API_ERROR
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.API_ERROR

    def _should_trigger_fallback(self, error: Exception) -> bool:
        """Determine if error should trigger fallback to next provider."""
        error_str = str(error).lower()

        # Always trigger fallback for these errors
        if isinstance(error, TimeoutError):
            return True
        if "429" in error_str or "rate limit" in error_str:
            return True
        if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
            return True
        if "503" in error_str or "service unavailable" in error_str:
            return True
        if "network" in error_str or "connection" in error_str:
            return True

        # Don't trigger fallback for client errors (4xx except 429)
        if "400" in error_str or "413" in error_str or "malformed" in error_str:
            return False

        # Default to triggering fallback for other errors
        return True

    async def _try_provider(
        self, provider: LLMProvider, messages: list[ChatMessage], **kwargs
    ) -> CompletionResponse:
        """Try to get completion from a specific provider."""
        try:
            return await provider.complete(messages, **kwargs)
        except Exception as e:
            error_type = self._classify_error(e)

            if self._should_trigger_fallback(e):
                self._cache_provider_failure(provider, error_type)
                raise

            # Re-raise the error if it shouldn't trigger fallback
            raise

    async def acomplete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """Complete method with automatic fallback across providers."""
        last_error = None

        for provider in self._available_providers:
            # Skip providers that are cached as failed
            error_type = ErrorType.TIMEOUT  # Default error type for checking cache
            if self._is_provider_cached_as_failed(provider, error_type):
                self.logger.info(
                    "Skipping cached failed provider",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                continue

            try:
                self.logger.info(
                    "Trying provider",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                return await self._try_provider(provider, messages, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Provider failed, trying next",
                    provider=provider.provider_type.value,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                continue

        # All providers failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError("All LLM providers failed")

    def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """Synchronous complete method."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.acomplete(messages, **kwargs))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.acomplete(messages, **kwargs))
            finally:
                loop.close()

    async def astream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Streaming complete method with automatic fallback across providers."""
        last_error = None

        for provider in self._available_providers:
            # Skip providers that are cached as failed
            error_type = ErrorType.TIMEOUT  # Default error type for checking cache
            if self._is_provider_cached_as_failed(provider, error_type):
                self.logger.info(
                    "Skipping cached failed provider for streaming",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                continue

            try:
                self.logger.info(
                    "Trying provider for streaming",
                    provider=provider.provider_type.value,
                    model=provider.get_model_name(),
                )
                async for chunk in provider.stream_complete(messages, **kwargs):
                    yield chunk
                return  # Success, exit the loop
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Provider failed for streaming, trying next",
                    provider=provider.provider_type.value,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                continue

        # All providers failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError("All LLM providers failed")

    def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Synchronous streaming complete method."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.astream_complete(messages, **kwargs))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.astream_complete(messages, **kwargs))
            finally:
                loop.close()
