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
import contextvars
import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

try:
    import redis  # type: ignore

    _HAS_REDIS = True
except Exception:  # pragma: no cover
    redis = None  # type: ignore
    _HAS_REDIS = False
try:
    from groq import AsyncGroq  # type: ignore

    _HAS_GROQ = True
except Exception:  # pragma: no cover
    AsyncGroq = None  # type: ignore
    _HAS_GROQ = False
from groq.types.chat import ChatCompletionChunk
from llama_index.core.base.llms.types import ChatMessage, CompletionResponse
from llama_index.core.llms import LLMMetadata
from llama_index.core.llms.custom import CustomLLM

try:
    from llama_index.llms.google_genai import GoogleGenAI as Gemini  # type: ignore

    _HAS_GEMINI = True
except Exception:  # pragma: no cover
    Gemini = None  # type: ignore
    _HAS_GEMINI = False
try:
    from llama_index.llms.openai import OpenAI  # type: ignore

    _HAS_OPENAI = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False
try:
    from llama_index.llms.openrouter import OpenRouter  # type: ignore

    _HAS_OPENROUTER = True
except Exception:  # pragma: no cover
    OpenRouter = None  # type: ignore
    _HAS_OPENROUTER = False
from pydantic import PrivateAttr

from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.metrics import VectorSearchMetrics

# LLM Router Implementation
#
# Features:
# - Priority-based routing: Gemini(OpenRouter) → Groq(Llama3) → ChatGPT → Local
# - Automatic error detection and provider fallback
# - Redis-based caching of failed providers (5-minute cooldown)
# - Comprehensive error handling and retry logic
# - Groq rate limit header parsing and quota sharing across workers
# - Prometheus metrics for monitoring provider health and rate limiting

logger = get_logger(__name__)

# Get metrics backend for provider monitoring
metrics = VectorSearchMetrics()

# Per-request model preference (used by LlamaIndex paths that don't pass model explicitly)
_model_preference_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_model_preference", default=None
)


class ProviderType(Enum):
    """Enumeration of supported LLM providers."""

    OPENROUTER = "openrouter"
    GROQ = "groq"
    OPENAI = "openai"
    GOOGLE = "google"
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

    def _ensure_messages(self, messages: list[ChatMessage] | str) -> list[ChatMessage]:
        """Normalize input into a list[ChatMessage].

        Some callers pass a formatted string instead of ChatMessage objects.
        Convert strings into a single user message to keep providers robust.
        """
        if isinstance(messages, list):
            return messages
        from llama_index.core.base.llms.types import MessageRole

        return [ChatMessage(role=MessageRole.USER, content=str(messages))]

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (has required credentials)."""
        pass

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        """Complete a conversation using the provider."""
        pass

    @abstractmethod
    async def stream_complete(
        self, messages: list[ChatMessage], **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Stream complete a conversation using the provider."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name for this provider."""
        pass


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation for accessing various models including Gemini."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(ProviderType.OPENROUTER, timeout_seconds)
        # Map common Gemini aliases to OpenRouter canonical IDs
        model_name = settings.LLM_MODEL_NAME
        alias_map = {
            "gemini-2.0-flash": "google/gemini-2.0-flash-001",
            "gemini-1.5-flash": "google/gemini-1.5-flash",
            "gemini-1.5-pro": "google/gemini-1.5-pro",
            "gemini-2.0-flash-thinking": "google/gemini-2.0-flash-thinking-exp",
        }
        self.model_name = alias_map.get(model_name, model_name)

        if self.is_available() and _HAS_OPENROUTER:
            self.client = OpenRouter(
                api_key=settings.OPENROUTER_API_KEY, model=self.model_name, timeout=timeout_seconds
            )

    def is_available(self) -> bool:
        return bool(settings.OPENROUTER_API_KEY) and _HAS_OPENROUTER

    async def complete(self, messages: list[ChatMessage], **kwargs) -> CompletionResponse:
        try:
            self.logger.info("Calling OpenRouter for completion", model=self.model_name)
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
            self.logger.info("Calling OpenRouter for streaming completion", model=self.model_name)
            async for response in self.client.astream_complete(messages, **kwargs):
                yield response
            self.logger.info("OpenRouter streaming completion successful")
        except Exception as e:
            self.logger.error(
                "OpenRouter streaming completion failed", error=str(e), error_type=type(e).__name__
            )
            raise

    def get_model_name(self) -> str:
        return self.model_name


class GroqProvider(LLMProvider):
    """Groq provider implementation with enhanced rate limit handling and quota sharing."""

    def __init__(self, timeout_seconds: int = None):
        # Use Groq-specific timeout if provided, otherwise use default
        timeout = timeout_seconds or settings.GROQ_TIMEOUT_SECONDS
        super().__init__(ProviderType.GROQ, timeout)
        self.model_name = settings.GROQ_MODEL_NAME

        if self.is_available() and _HAS_GROQ:
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

        # Initialize Redis client for quota sharing
        try:
            from app.core.redis import redis_client

            self.redis_client = redis_client
        except Exception as e:
            self.logger.warning("Failed to initialize Redis for Groq quota sharing", error=str(e))
            self.redis_client = None

    def is_available(self) -> bool:
        return bool(settings.GROQ_API_KEY) and _HAS_GROQ

    def _parse_rate_limit_headers(self, response_headers: dict) -> dict:
        """Parse Groq rate limit headers into structured data."""
        rate_limit_info = {}

        # Parse rate limit headers
        headers_map = {
            "x-ratelimit-limit-requests": "limit_requests_per_day",
            "x-ratelimit-limit-tokens": "limit_tokens_per_minute",
            "x-ratelimit-remaining-requests": "remaining_requests",
            "x-ratelimit-remaining-tokens": "remaining_tokens",
            "x-ratelimit-reset-requests": "reset_requests",
            "x-ratelimit-reset-tokens": "reset_tokens",
            "retry-after": "retry_after_seconds",
        }

        for header_name, field_name in headers_map.items():
            if header_name in response_headers:
                value = response_headers[header_name]
                try:
                    # Convert to appropriate type
                    if "remaining" in field_name or "limit" in field_name:
                        rate_limit_info[field_name] = int(value)
                    elif "retry_after" in field_name:
                        rate_limit_info[field_name] = int(value)
                    else:
                        # Parse reset times - could be in seconds or ISO format
                        if value.endswith("s"):
                            rate_limit_info[field_name] = int(value[:-1])
                        elif value.endswith("ms"):
                            rate_limit_info[field_name] = int(value[:-2]) / 1000
                        else:
                            rate_limit_info[field_name] = int(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Failed to parse rate limit header {header_name}: {value}")

        # Add timestamp when this info was captured
        rate_limit_info["timestamp"] = time.time()

        return rate_limit_info

    def _store_rate_limit_info(self, rate_limit_info: dict):
        """Store rate limit information in Redis for sharing across workers."""
        if not self.redis_client or not rate_limit_info:
            return

        try:
            # Store with 5-minute expiry (should be refreshed by new requests)
            redis_key = "groq:rate_limit"
            self.redis_client.setex(
                redis_key,
                300,  # 5 minutes TTL
                json.dumps(rate_limit_info),
            )
            self.logger.debug("Stored Groq rate limit info in Redis", data=rate_limit_info)
        except Exception as e:
            self.logger.warning("Failed to store rate limit info in Redis", error=str(e))

    def _get_stored_rate_limit_info(self) -> dict:
        """Retrieve stored rate limit information from Redis."""
        if not self.redis_client:
            return {}

        try:
            redis_key = "groq:rate_limit"
            stored_data = self.redis_client.get(redis_key)
            if stored_data:
                return json.loads(stored_data)
        except Exception as e:
            self.logger.warning("Failed to retrieve rate limit info from Redis", error=str(e))

        return {}

    def _should_skip_due_to_rate_limits(self) -> tuple[bool, int]:
        """Check if we should skip this provider due to known rate limits."""
        stored_info = self._get_stored_rate_limit_info()
        if not stored_info:
            return False, 0

        current_time = time.time()
        info_age = current_time - stored_info.get("timestamp", 0)

        # If info is older than 2 minutes, consider it stale
        if info_age > 120:
            return False, 0

        # Check if we're close to hitting limits
        remaining_requests = stored_info.get("remaining_requests", float("inf"))
        remaining_tokens = stored_info.get("remaining_tokens", float("inf"))

        # Skip if we have very few requests/tokens left (conservative approach)
        if remaining_requests <= 2 or remaining_tokens <= 1000:
            retry_after = stored_info.get("retry_after_seconds", 60)
            self.logger.warning(
                "Skipping Groq due to low quota",
                remaining_requests=remaining_requests,
                remaining_tokens=remaining_tokens,
                retry_after=retry_after,
            )
            # Increment rate limit skip metric
            metrics.increment_counter(
                "llm_provider_rate_limit_hits_total",
                {"provider": "groq", "action": "preemptive_skip"},
            )
            return True, retry_after

        return False, 0

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

    def _add_jitter(self, base_delay: float) -> float:
        """Add jitter to delay to avoid thundering herd effects."""
        # Add ±25% jitter
        jitter_factor = 0.75 + (random.random() * 0.5)  # 0.75 to 1.25
        return base_delay * jitter_factor

    def _handle_successful_response(self, result):
        """Handle successful response and extract rate limit headers."""
        if hasattr(result, "_raw_response") and hasattr(result._raw_response, "headers"):
            headers = dict(result._raw_response.headers)
            rate_limit_info = self._parse_rate_limit_headers(headers)
            if rate_limit_info:
                self._store_rate_limit_info(rate_limit_info)
        return result

    def _handle_timeout_error(self, attempt: int, max_retries: int, error: Exception):
        """Handle timeout errors with appropriate logging and retry logic."""
        if attempt == max_retries - 1:
            raise TimeoutError(f"Groq request timed out after {self.timeout_seconds}s") from error

        wait_time = self._add_jitter(2**attempt)
        self.logger.warning(
            f"Groq request timed out, retrying in {wait_time:.1f}s",
            attempt=attempt + 1,
            timeout=self.timeout_seconds,
        )
        return wait_time

    def _handle_rate_limit_error(
        self, attempt: int, max_retries: int, error: Exception, error_str: str
    ):
        """Handle rate limiting and capacity errors with retry logic."""
        # Increment rate limit hit metric
        metrics.increment_counter(
            "llm_provider_rate_limit_hits_total", {"provider": "groq", "action": "hit"}
        )

        # Try to parse retry-after from error or use exponential backoff
        retry_after = None
        if hasattr(error, "response") and hasattr(error.response, "headers"):
            retry_after_header = error.response.headers.get("retry-after")
            if retry_after_header:
                try:
                    retry_after = int(retry_after_header)
                except ValueError:
                    pass

        if attempt == max_retries - 1:
            # Store rate limit info before failing
            if retry_after:
                rate_limit_info = {
                    "retry_after_seconds": retry_after,
                    "remaining_requests": 0,
                    "remaining_tokens": 0,
                    "timestamp": time.time(),
                }
                self._store_rate_limit_info(rate_limit_info)
            raise

        # Use retry-after if provided, otherwise exponential backoff
        if retry_after:
            wait_time = min(retry_after, 60)  # Cap at 60 seconds
        else:
            wait_time = self._add_jitter(2**attempt)

        self.logger.warning(
            f"Groq rate limited, retrying in {wait_time:.1f}s",
            attempt=attempt + 1,
            retry_after=retry_after,
            error_code="429" if "429" in error_str else "498",
        )
        return wait_time

    def _handle_service_error(self, attempt: int, max_retries: int, error_str: str):
        """Handle service unavailable errors."""
        if attempt == max_retries - 1:
            raise

        wait_time = self._add_jitter(2**attempt)
        self.logger.warning(
            f"Groq service unavailable, retrying in {wait_time:.1f}s",
            attempt=attempt + 1,
        )
        return wait_time

    def _check_rate_limit_preemption(self, attempt: int):
        """Check if we should skip due to known rate limits."""
        should_skip, suggested_delay = self._should_skip_due_to_rate_limits()
        if should_skip and attempt == 0:  # Only skip on first attempt
            wait_time = min(suggested_delay, 30)  # Cap at 30 seconds
            raise Exception(f"Rate limit preemption: retry after {wait_time}s")

    async def _execute_with_timeout(self, func, *args, **kwargs):
        """Execute the function with timeout and handle successful response."""
        result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout_seconds)
        return self._handle_successful_response(result)

    async def _handle_error(self, e: Exception, attempt: int, max_retries: int) -> float:
        """Handle various error types and return wait time if retry is needed."""
        error_str = str(e).lower()

        # Don't retry on rate limit preemption - it's an immediate skip
        if "rate limit preemption" in error_str:
            raise

        # Handle rate limiting (429) and capacity exceeded (498)
        if (
            "429" in error_str
            or "rate limit" in error_str
            or "498" in error_str
            or "capacity" in error_str
        ):
            return self._handle_rate_limit_error(attempt, max_retries, e, error_str)

        # Handle service unavailable errors
        if "503" in error_str or "service unavailable" in error_str:
            return self._handle_service_error(attempt, max_retries, error_str)

        # For other errors, don't retry
        raise

    async def _retry_with_backoff(self, func, *args, max_retries=3, **kwargs):
        """Enhanced retry function with exponential backoff, jitter, and rate limit handling."""
        for attempt in range(max_retries):
            try:
                # Check if we should skip due to known rate limits
                self._check_rate_limit_preemption(attempt)
                return await self._execute_with_timeout(func, *args, **kwargs)

            except asyncio.TimeoutError as err:
                wait_time = self._handle_timeout_error(attempt, max_retries, err)
                await asyncio.sleep(wait_time)

            except Exception as e:
                wait_time = await self._handle_error(e, attempt, max_retries)
                await asyncio.sleep(wait_time)

    def _ensure_messages(self, messages: list[ChatMessage] | str) -> list[ChatMessage]:
        """Normalize input into a list[ChatMessage] for providers.

        LlamaIndex sometimes calls CustomLLM.complete with a formatted string
        when `formatted=True` is passed, which breaks providers that expect
        ChatMessage instances. This helper converts a string into a single
        user ChatMessage.
        """
        if isinstance(messages, list):
            return messages
        from llama_index.core.base.llms.types import MessageRole

        return [ChatMessage(role=MessageRole.USER, content=str(messages))]

    async def complete(self, messages: list[ChatMessage] | str, **kwargs) -> CompletionResponse:
        messages = self._ensure_messages(messages)
        try:
            self.logger.info("Calling Groq for completion", model=self.model_name)

            # Convert messages to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages)

            # Prepare parameters for Groq API
            # Base required params
            groq_params: dict = {
                "model": self.model_name,
                "messages": groq_messages,
            }

            # If caller still sends deprecated ``max_tokens`` keep backward-compatibility
            if "max_completion_tokens" in kwargs:
                groq_params["max_completion_tokens"] = kwargs["max_completion_tokens"]
            elif "max_tokens" in kwargs:
                # Transparently migrate deprecated param
                groq_params["max_completion_tokens"] = kwargs["max_tokens"]
            else:
                # Sensible default when nothing supplied
                groq_params["max_completion_tokens"] = 1024

            # Provide default temperature if caller omitted it
            groq_params["temperature"] = kwargs.get("temperature", 0.7)

            # Forward any other recognised Groq parameters directly – this keeps provider
            # implementation future-proof without having to whitelist each one manually.
            allowed_extra_params = [
                "top_p",
                "seed",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "tool_choice",
                "tools",
                "parallel_tool_calls",
                "response_format",
                "search_settings",
                "user",
                "stream",  # although not used on non-stream request, we allow it if passed
            ]
            for key in allowed_extra_params:
                if key in kwargs and key not in groq_params:
                    groq_params[key] = kwargs[key]

            # Make the API call with enhanced retry logic
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
        self, messages: list[ChatMessage] | str, **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        messages = self._ensure_messages(messages)
        try:
            self.logger.info("Calling Groq for streaming completion", model=self.model_name)

            # Convert messages to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages)

            # Prepare parameters for Groq API
            groq_params: dict = {
                "model": self.model_name,
                "messages": groq_messages,
                "stream": True,
            }

            # Token limits
            if "max_completion_tokens" in kwargs:
                groq_params["max_completion_tokens"] = kwargs["max_completion_tokens"]
            elif "max_tokens" in kwargs:
                groq_params["max_completion_tokens"] = kwargs["max_tokens"]
            else:
                groq_params["max_completion_tokens"] = 1024

            groq_params["temperature"] = kwargs.get("temperature", 0.7)

            # Forward recognised extras
            allowed_extra_params = [
                "top_p",
                "seed",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "tool_choice",
                "tools",
                "parallel_tool_calls",
                "response_format",
                "search_settings",
                "user",
            ]
            for key in allowed_extra_params:
                if key in kwargs and key not in groq_params:
                    groq_params[key] = kwargs[key]

            # Make the streaming API call
            async with asyncio.timeout(self.timeout_seconds):
                stream = self.client.chat.completions.create(**groq_params)

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
        if self.is_available() and _HAS_OPENAI:
            self.client = OpenAI(
                api_key=settings.OPENAI_API_KEY, model=self.model_name, timeout=timeout_seconds
            )

    def is_available(self) -> bool:
        return bool(settings.OPENAI_API_KEY) and _HAS_OPENAI

    async def complete(self, messages: list[ChatMessage] | str, **kwargs) -> CompletionResponse:
        messages = self._ensure_messages(messages)
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
        self, messages: list[ChatMessage] | str, **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        messages = self._ensure_messages(messages)
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


class GoogleAIStudioProvider(LLMProvider):
    """Google AI Studio (Gemini) provider using LlamaIndex Gemini LLM."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(ProviderType.GOOGLE, timeout_seconds)
        self.model_name = settings.GOOGLE_MODEL_NAME
        # Only construct client when both key and model name are valid strings
        if self.is_available() and _HAS_GEMINI:
            try:
                self.client = Gemini(
                    api_key=settings.GOOGLE_AI_STUDIO_API_KEY, model=self.model_name
                )
            except Exception as e:
                self.logger.warning("Failed to initialize Google Gemini client", error=str(e))
                self.client = None

    def is_available(self) -> bool:
        return (
            _HAS_GEMINI
            and isinstance(settings.GOOGLE_AI_STUDIO_API_KEY, str)
            and bool(settings.GOOGLE_AI_STUDIO_API_KEY)
            and isinstance(self.model_name, str)
        )

    async def complete(self, messages: list[ChatMessage] | str, **kwargs) -> CompletionResponse:
        messages = self._ensure_messages(messages)
        try:
            self.logger.info("Calling Google Gemini for completion", model=self.model_name)
            response = await asyncio.wait_for(
                self.client.acomplete(messages, **kwargs), timeout=self.timeout_seconds
            )
            self.logger.info("Google Gemini completion successful")
            return response
        except asyncio.TimeoutError as err:
            self.logger.warning("Google Gemini request timed out", timeout=self.timeout_seconds)
            raise TimeoutError(
                f"Google Gemini request timed out after {self.timeout_seconds}s"
            ) from err
        except Exception as e:
            self.logger.error(
                "Google Gemini completion failed", error=str(e), error_type=type(e).__name__
            )
            raise

    async def stream_complete(
        self, messages: list[ChatMessage] | str, **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        messages = self._ensure_messages(messages)
        try:
            self.logger.info(
                "Calling Google Gemini for streaming completion", model=self.model_name
            )
            async for chunk in self.client.astream_complete(messages, **kwargs):
                yield chunk
            self.logger.info("Google Gemini streaming completion successful")
        except asyncio.TimeoutError as err:
            self.logger.warning(
                "Google Gemini streaming request timed out", timeout=self.timeout_seconds
            )
            raise TimeoutError(
                f"Google Gemini streaming request timed out after {self.timeout_seconds}s"
            ) from err
        except Exception as e:
            self.logger.error(
                "Google Gemini streaming completion failed",
                error=str(e),
                error_type=type(e).__name__,
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
            if _HAS_REDIS:
                self._redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self._redis_client.ping()  # Test connection
                logger.info("Redis connection established for LLM Router")
            else:
                self._redis_client = None
        except Exception as e:
            logger.error("Failed to connect to Redis for LLM Router", error=str(e))
            self._redis_client = None

        # Initialize providers in priority order
        # Prefer Google AI Studio by default; then fall back to others
        self._providers = [
            GoogleAIStudioProvider(settings.LLM_TIMEOUT_SECONDS),
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

    # LlamaIndex hook: capture model preference from external callers via context var
    def set_model_preference(self, preferred_model: str | None) -> None:
        _model_preference_ctx.set(preferred_model)

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

    def _apply_model_preference(self, preferred_model: str | None) -> list[LLMProvider]:
        """Reorder or filter providers based on preferred model alias.

        Simple mapping:
        - gemini* -> prioritize Google/OpenRouter
        - llama3* -> prioritize Groq
        - gpt-*   -> prioritize OpenAI
        """
        if not preferred_model:
            return self._available_providers

        m = preferred_model.lower()
        # Buckets
        google_like = []
        groq_like = []
        openai_like = []
        others = []
        for p in self._available_providers:
            if p.provider_type in (ProviderType.GOOGLE, ProviderType.OPENROUTER):
                google_like.append(p)
            elif p.provider_type == ProviderType.GROQ:
                groq_like.append(p)
            elif p.provider_type == ProviderType.OPENAI:
                openai_like.append(p)
            else:
                others.append(p)

        if m.startswith("gemini") or m.startswith("google"):
            return google_like + groq_like + openai_like + others
        if m.startswith("llama3"):
            return groq_like + google_like + openai_like + others
        if m.startswith("gpt-") or m.startswith("chatgpt"):
            return openai_like + google_like + groq_like + others
        return self._available_providers

    async def acomplete(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs
    ) -> CompletionResponse:
        """Complete method with automatic fallback across providers."""
        last_error = None

        preferred = model or _model_preference_ctx.get()
        provider_order = self._apply_model_preference(preferred)
        for provider in provider_order:
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
        self, messages: list[ChatMessage], model: str | None = None, **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Streaming complete method with automatic fallback across providers."""
        last_error = None

        preferred = model or _model_preference_ctx.get()
        provider_order = self._apply_model_preference(preferred)
        for provider in provider_order:
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
