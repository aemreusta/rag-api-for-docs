"""
Enhanced embedding manager with quota management and provider fallback.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

try:
    import redis  # type: ignore

    _HAS_REDIS = True
except Exception:  # pragma: no cover
    redis = None  # type: ignore
    _HAS_REDIS = False
from llama_index.core.base.embeddings.base import BaseEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.embeddings import LoggedEmbeddingWrapper
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ProviderStatus(str, Enum):
    """Embedding provider status."""

    ACTIVE = "active"
    QUOTA_EXHAUSTED = "quota_exhausted"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class QuotaError(Exception):
    """Exception raised when provider quota is exhausted."""

    pass


class InMemoryRedis:
    """Minimal in-memory stub for Redis used in tests when redis is unavailable.

    Supports only the methods used by this module: get, setex, delete, keys.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def setex(self, key: str, _ttl_seconds: int, value: str) -> None:
        self._store[key] = value

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                deleted += 1
        return deleted

    def keys(self, pattern: str) -> list[str]:
        # Very simple prefix pattern support like "embedding_quota:*"
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._store if k.startswith(prefix)]
        return [k for k in self._store if k == pattern]


class EmbeddingManager:
    """
    Manages multiple embedding providers with quota tracking and intelligent fallback.
    """

    QUOTA_RESET_TIMES = {
        "google": 24 * 3600,  # Google resets daily
        "qwen": None,  # No quota limits for local deployment
        "huggingface": 3600,  # HF resets hourly for free tier
    }

    def __init__(self, redis_client: Any | None = None):
        self.redis_client: Any = redis_client or self._init_redis()
        self.providers: dict[str, BaseEmbedding] = {}
        self.provider_status: dict[str, ProviderStatus] = {}
        self.last_quota_reset: dict[str, datetime] = {}

        # Initialize providers based on configuration
        self._initialize_providers()

    def _init_redis(self) -> Any:
        """Initialize Redis connection for quota state tracking.

        Falls back to an in-memory stub when redis is not installed. This keeps
        tests hermetic and avoids import errors.
        """
        if _HAS_REDIS:
            return redis.Redis(  # type: ignore[attr-defined]
                host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True
            )
        logger.warning("Redis module not available; using in-memory quota store for tests")
        return InMemoryRedis()

    def _initialize_providers(self):
        """Initialize all configured embedding providers."""
        # Primary provider (Google Gemini)
        if settings.EMBEDDING_PROVIDER == "google" and settings.GOOGLE_AI_STUDIO_API_KEY:
            try:
                self.providers["google"] = self._create_google_provider()
                self.provider_status["google"] = ProviderStatus.ACTIVE
                logger.info("Google embedding provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google provider: {e}")
                self.provider_status["google"] = ProviderStatus.ERROR

        # HuggingFace Embedding Service fallback provider
        embedding_service_endpoint = settings.EMBEDDING_SERVICE_ENDPOINT or settings.QWEN_ENDPOINT
        if embedding_service_endpoint:
            try:
                self.providers["qwen"] = self._create_qwen_provider()
                self.provider_status["qwen"] = ProviderStatus.ACTIVE
                logger.info("Qwen embedding provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Qwen provider: {e}")
                self.provider_status["qwen"] = ProviderStatus.ERROR

        # HuggingFace fallback
        try:
            self.providers["huggingface"] = self._create_huggingface_provider()
            self.provider_status["huggingface"] = ProviderStatus.ACTIVE
            logger.info("HuggingFace embedding provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace provider: {e}")
            self.provider_status["huggingface"] = ProviderStatus.ERROR

    def _create_google_provider(self) -> BaseEmbedding:
        """Create Google Gemini embedding provider."""
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        model_id = settings.EMBEDDING_MODEL_NAME
        if not (model_id.startswith("models/") or model_id.startswith("tunedModels/")):
            model_id = f"models/{model_id}"

        return LoggedEmbeddingWrapper(
            GoogleGenAIEmbedding(
                model_name=model_id,
                api_key=settings.GOOGLE_AI_STUDIO_API_KEY,
                task_type="retrieval_document",
            ),
            provider="google",
            model_name=model_id,
        )

    def _create_qwen_provider(self) -> BaseEmbedding:
        """Create embedding service provider using OpenAI-compatible API."""
        from app.core.qwen_openai_embedding import QwenOpenAIEmbedding

        # Use new settings with backward compatibility
        endpoint = settings.EMBEDDING_SERVICE_ENDPOINT or settings.QWEN_ENDPOINT
        model_name = settings.EMBEDDING_SERVICE_MODEL_NAME or "qwen3-embedding-0.6b"
        dimensions = settings.EMBEDDING_SERVICE_DIMENSIONS or settings.QWEN_DIMENSIONS

        return LoggedEmbeddingWrapper(
            QwenOpenAIEmbedding(
                api_base=endpoint,
                api_key="embedding-service-key",
                model_name=model_name,
                dimensions=dimensions,
            ),
            provider="embedding_service",
            model_name=model_name,
        )

    def _create_huggingface_provider(self) -> BaseEmbedding:
        """Create HuggingFace embedding provider."""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return LoggedEmbeddingWrapper(
            HuggingFaceEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            provider="huggingface",
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )

    def _check_quota_status(self, provider: str) -> ProviderStatus:
        """Check if provider quota is available."""
        quota_key = f"embedding_quota:{provider}"

        # Check Redis for quota status
        quota_data = self.redis_client.get(quota_key)
        if quota_data:
            data = json.loads(quota_data)
            quota_reset_time = datetime.fromisoformat(data["reset_time"])

            if datetime.utcnow() < quota_reset_time:
                return ProviderStatus.QUOTA_EXHAUSTED
            else:
                # Quota should have reset, clear the key
                self.redis_client.delete(quota_key)

        return self.provider_status.get(provider, ProviderStatus.ERROR)

    def _mark_quota_exhausted(self, provider: str):
        """Mark provider quota as exhausted with reset time."""
        reset_time = datetime.utcnow() + timedelta(
            seconds=self.QUOTA_RESET_TIMES.get(provider, 3600)
        )
        quota_key = f"embedding_quota:{provider}"

        quota_data = {
            "exhausted_at": datetime.utcnow().isoformat(),
            "reset_time": reset_time.isoformat(),
            "provider": provider,
        }

        # Store with expiration
        self.redis_client.setex(
            quota_key, self.QUOTA_RESET_TIMES.get(provider, 3600), json.dumps(quota_data)
        )

        self.provider_status[provider] = ProviderStatus.QUOTA_EXHAUSTED

        logger.warning(
            "Provider quota exhausted",
            provider=provider,
            reset_time=reset_time.isoformat(),
            next_check_seconds=self.QUOTA_RESET_TIMES.get(provider, 3600),
        )

    def _get_available_provider(self) -> str | None:
        """Get the next available provider based on priority and quota status."""
        # Provider priority order
        provider_priority = ["google", "qwen", "huggingface"]

        for provider in provider_priority:
            if provider not in self.providers:
                continue

            status = self._check_quota_status(provider)
            if status == ProviderStatus.ACTIVE:
                return provider

        return None

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error indicates quota exhaustion."""
        error_str = str(error).lower()
        quota_indicators = ["quota", "rate limit", "resource exhausted", "429", "too many requests"]

        return any(indicator in error_str for indicator in quota_indicators)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts with provider fallback.
        """
        provider = self._get_available_provider()

        if not provider:
            # All providers exhausted, calculate wait time
            min_wait = self._calculate_min_wait_time()
            raise QuotaError(
                f"All embedding providers quota exhausted. Minimum wait time: {min_wait} seconds"
            )

        try:
            embedding_model = self.providers[provider]

            logger.info(
                "Generating embeddings",
                provider=provider,
                text_count=len(texts),
                total_chars=sum(len(text) for text in texts),
            )

            start_time = time.time()
            embeddings = embedding_model.get_text_embeddings(texts)
            duration = time.time() - start_time

            logger.info(
                "Embeddings generated successfully",
                provider=provider,
                text_count=len(texts),
                embedding_dimensions=len(embeddings[0]) if embeddings else 0,
                duration_seconds=round(duration, 2),
            )

            return embeddings

        except Exception as e:
            if self._is_quota_error(e):
                logger.warning(
                    "Provider quota exhausted during embedding generation",
                    provider=provider,
                    error=str(e),
                )
                self._mark_quota_exhausted(provider)
                # Retry with different provider
                raise QuotaError(f"Quota exhausted for {provider}: {str(e)}") from e
            else:
                logger.error(
                    "Embedding generation failed",
                    provider=provider,
                    error=str(e),
                    text_count=len(texts),
                )
                raise

    def get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for single text."""
        embeddings = self.get_text_embeddings([text])
        return embeddings[0]

    def _calculate_min_wait_time(self) -> int:
        """Calculate minimum wait time until any provider becomes available."""
        min_wait = float("inf")

        for provider in self.providers.keys():
            quota_key = f"embedding_quota:{provider}"
            quota_data = self.redis_client.get(quota_key)

            if quota_data:
                data = json.loads(quota_data)
                reset_time = datetime.fromisoformat(data["reset_time"])
                wait_time = (reset_time - datetime.utcnow()).total_seconds()

                if wait_time > 0 and wait_time < min_wait:
                    min_wait = wait_time

        return int(min_wait) if min_wait != float("inf") else 0

    def get_provider_status(self) -> dict[str, dict]:
        """Get current status of all providers."""
        status = {}

        for provider in self.providers.keys():
            current_status = self._check_quota_status(provider)
            status[provider] = {
                "status": current_status.value,
                "available": current_status == ProviderStatus.ACTIVE,
            }

            # Add quota info if exhausted
            if current_status == ProviderStatus.QUOTA_EXHAUSTED:
                quota_key = f"embedding_quota:{provider}"
                quota_data = self.redis_client.get(quota_key)
                if quota_data:
                    data = json.loads(quota_data)
                    status[provider]["quota_reset_time"] = data["reset_time"]
                    reset_time = datetime.fromisoformat(data["reset_time"])
                    status[provider]["wait_seconds"] = max(
                        0, int((reset_time - datetime.utcnow()).total_seconds())
                    )

        return status


# Global embedding manager instance
_embedding_manager: EmbeddingManager | None = None


def get_embedding_manager() -> EmbeddingManager:
    """Get or create global embedding manager instance."""
    global _embedding_manager

    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()

    return _embedding_manager


def reset_embedding_manager():
    """Reset global embedding manager (useful for testing)."""
    global _embedding_manager
    _embedding_manager = None
