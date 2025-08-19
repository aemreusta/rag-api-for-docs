from __future__ import annotations

import os
import time
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.request_tracking import get_request_tracker

logger = get_logger(__name__)


class LoggedEmbeddingWrapper(BaseEmbedding):
    """Wrapper that adds structured logging to embedding API requests."""

    def __init__(self, embedding_model: BaseEmbedding, provider: str, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self._embedding = embedding_model
        self._provider = provider
        self._model_name = model_name
        self._logger = get_logger(f"{__name__}.{provider}")
        self._tracker = get_request_tracker()

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query with logging."""
        return self.get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version of query embedding."""
        if hasattr(self._embedding, "_aget_query_embedding"):
            start_time = time.time()
            try:
                embedding = await self._embedding._aget_query_embedding(query)
                duration_ms = round((time.time() - start_time) * 1000, 2)

                self._logger.info(
                    "Async embedding API request completed",
                    provider=self._provider,
                    model=self._model_name,
                    input_size_chars=len(query),
                    input_size_tokens=self._estimate_tokens(query),
                    response_time_ms=duration_ms,
                    output_dimensions=len(embedding) if embedding else 0,
                    request_type="async_query",
                )

                return embedding
            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)

                self._logger.error(
                    "Async embedding API request failed",
                    provider=self._provider,
                    model=self._model_name,
                    input_size_chars=len(query),
                    response_time_ms=duration_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    request_type="async_query",
                )
                raise
        else:
            # Fallback to sync version
            return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text with comprehensive tracking."""
        with self._tracker.track_embedding_request(
            provider=self._provider, model=self._model_name, input_type="text", batch_size=1
        ) as context:
            embedding = self._embedding.get_text_embedding(text)

            # Add detailed metrics to context
            context.update(
                {
                    "input_size_chars": len(text),
                    "input_size_tokens": self._estimate_tokens(text),
                    "output_dimensions": len(embedding) if embedding else 0,
                }
            )

            return embedding

    def get_text_embedding(self, text: str) -> list[float]:
        """Public interface for getting text embeddings."""
        return self._get_text_embedding(text)

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts with logging."""
        start_time = time.time()
        batch_size = len(texts)
        total_chars = sum(len(text) for text in texts)

        try:
            embeddings = self._embedding.get_text_embedding_batch(texts)
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.info(
                "Embedding API batch request completed",
                provider=self._provider,
                model=self._model_name,
                batch_size=batch_size,
                total_input_size_chars=total_chars,
                avg_input_size_chars=round(total_chars / batch_size, 1),
                estimated_tokens=sum(self._estimate_tokens(text) for text in texts),
                response_time_ms=duration_ms,
                output_dimensions=len(embeddings[0]) if embeddings and embeddings[0] else 0,
                request_type="batch",
            )

            return embeddings
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.error(
                "Embedding API batch request failed",
                provider=self._provider,
                model=self._model_name,
                batch_size=batch_size,
                total_input_size_chars=total_chars,
                response_time_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
                request_type="batch",
            )
            raise

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars = 1 token average for English)."""
        return len(text) // 4

    def __getattr__(self, name):
        """Forward other attributes to the wrapped embedding model."""
        return getattr(self._embedding, name)


def _get_openai_embedding(model_name: str) -> Any:
    """Initialize OpenAI embedding with validation."""
    start_time = time.time()

    # Validate OpenAI API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        error_msg = (
            "OpenAI embedding provider selected but no API key found. "
            "Please set OPENAI_API_KEY in your environment."
        )
        logger.error(
            "OpenAI embedding initialization failed",
            provider="openai",
            model=model_name,
            error="missing_api_key",
        )
        raise ValueError(error_msg)

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding

        embedding = OpenAIEmbedding(model=model_name)

        logger.info(
            "Embedding provider initialized",
            provider="openai",
            model=model_name,
            initialization_time_ms=round((time.time() - start_time) * 1000, 2),
        )

        return embedding
    except ImportError as e:
        error_msg = (
            f"OpenAI embedding provider selected but required dependencies not available: {e}. "
            "Please install openai or llama-index[openai] package."
        )
        logger.error(
            "OpenAI embedding initialization failed",
            provider="openai",
            model=model_name,
            error="missing_dependencies",
            error_details=str(e),
        )
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to initialize OpenAI embedding provider: {e}. "
            "Please check your OPENAI_API_KEY and network connectivity."
        )
        logger.error(
            "OpenAI embedding initialization failed",
            provider="openai",
            model=model_name,
            error="initialization_error",
            error_details=str(e),
        )
        raise RuntimeError(error_msg) from e


def _get_google_embedding(provider: str, model_name: str) -> Any:
    """Initialize Google embedding with validation."""
    start_time = time.time()

    # Validate Google API key is available before attempting to use Google provider
    google_api_key = settings.GOOGLE_AI_STUDIO_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        error_msg = (
            f"Google embedding provider selected ('{provider}') but no API key found. "
            "Please set GOOGLE_AI_STUDIO_API_KEY in your environment or configuration."
        )
        logger.error(
            "Google embedding initialization failed",
            provider=provider,
            model=model_name,
            error="missing_api_key",
        )
        raise ValueError(error_msg)

    try:
        # Try to use the newer Google GenAI embedding first
        try:
            from llama_index.embeddings.google_genai import GoogleGenerativeAIEmbedding

            # Set the API key in environment if not already set
            if settings.GOOGLE_AI_STUDIO_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_AI_STUDIO_API_KEY

            model_id = model_name
            if not (model_id.startswith("models/") or model_id.startswith("tunedModels/")):
                model_id = f"models/{model_id}"

            # Initialize with the newer GoogleGenerativeAIEmbedding
            embedding = GoogleGenerativeAIEmbedding(
                model_name=model_id, api_key=google_api_key, task_type="retrieval_document"
            )

            logger.info(
                "Embedding provider initialized",
                provider="google_genai",
                model=model_id,
                initialization_time_ms=round((time.time() - start_time) * 1000, 2),
            )

            return embedding
        except ImportError:
            # Fallback to the deprecated GeminiEmbedding if newer one is not available
            from llama_index.embeddings.google import GeminiEmbedding

            # Set the API key in environment if not already set
            if settings.GOOGLE_AI_STUDIO_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_AI_STUDIO_API_KEY

            model_id = model_name
            if not (model_id.startswith("models/") or model_id.startswith("tunedModels/")):
                model_id = f"models/{model_id}"

            # Initialize GeminiEmbedding with correct parameters
            embedding = GeminiEmbedding(
                model_name=model_id, api_key=google_api_key, task_type="retrieval_document"
            )

            logger.info(
                "Embedding provider initialized",
                provider="google_legacy",
                model=model_id,
                initialization_time_ms=round((time.time() - start_time) * 1000, 2),
            )

            return embedding
    except ImportError as e:
        error_msg = (
            f"Google embedding provider selected but required dependencies not available: {e}. "
            "Please install google-generativeai or llama-index[google] package."
        )
        logger.error(
            "Google embedding initialization failed",
            provider=provider,
            model=model_name,
            error="missing_dependencies",
            error_details=str(e),
        )
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to initialize Google embedding provider: {e}. "
            "Please check your GOOGLE_AI_STUDIO_API_KEY and network connectivity."
        )
        logger.error(
            "Google embedding initialization failed",
            provider=provider,
            model=model_name,
            error="initialization_error",
            error_details=str(e),
        )
        raise RuntimeError(error_msg) from e


def get_embedding_model() -> Any:
    """Get embedding model based on configured provider with strict validation."""
    provider = (settings.EMBEDDING_PROVIDER or "hf").lower()
    model_name = settings.EMBEDDING_MODEL_NAME
    start_time = time.time()

    logger.info("Embedding model selection started", provider=provider, model=model_name)

    if provider == "openai":
        try:
            embedding = _get_openai_embedding(model_name)
            return LoggedEmbeddingWrapper(embedding, "openai", model_name)
        except (ImportError, ValueError, RuntimeError) as e:
            # Fall back to HF on any error
            logger.warning(
                "Embedding provider fallback to HuggingFace",
                original_provider="openai",
                fallback_provider="huggingface",
                fallback_model=model_name,
                error=str(e),
            )
            embedding = HuggingFaceEmbedding(model_name=model_name)
            logger.info(
                "Embedding provider initialized",
                provider="huggingface",
                model=model_name,
                initialization_time_ms=round((time.time() - start_time) * 1000, 2),
            )
            return LoggedEmbeddingWrapper(embedding, "huggingface", model_name)
    elif provider in ("google", "gemini", "google_genai"):
        try:
            embedding = _get_google_embedding(provider, model_name)
            return LoggedEmbeddingWrapper(embedding, provider, model_name)
        except (ImportError, ValueError, RuntimeError) as e:
            # Fall back to HF on any error
            logger.warning(
                "Embedding provider fallback to HuggingFace",
                original_provider=provider,
                fallback_provider="huggingface",
                fallback_model=model_name,
                error=str(e),
            )
            embedding = HuggingFaceEmbedding(model_name=model_name)
            logger.info(
                "Embedding provider initialized",
                provider="huggingface",
                model=model_name,
                initialization_time_ms=round((time.time() - start_time) * 1000, 2),
            )
            return LoggedEmbeddingWrapper(embedding, "huggingface", model_name)

    # Default HF fallback
    embedding = HuggingFaceEmbedding(model_name=model_name)
    logger.info(
        "Embedding provider initialized",
        provider="huggingface",
        model=model_name,
        initialization_time_ms=round((time.time() - start_time) * 1000, 2),
    )
    return LoggedEmbeddingWrapper(embedding, "huggingface", model_name)
