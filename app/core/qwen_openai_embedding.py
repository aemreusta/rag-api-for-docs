"""
OpenAI-compatible Qwen3-Embedding client using vLLM server.

This replaces the custom HTTP client with a standard OpenAI client
for better compatibility and easier model switching in the future.
"""

from __future__ import annotations

import time
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from openai import OpenAI

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class QwenOpenAIEmbedding(BaseEmbedding):
    """Qwen3-Embedding implementation using OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str,
        api_key: str = "qwen-embedding-key",
        model_name: str = "qwen3-embedding-0.6b",
        dimensions: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, embed_batch_size=batch_size, **kwargs)

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=f"{self.api_base}/v1",
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(
            "Qwen OpenAI embedding client initialized",
            api_base=self.api_base,
            model_name=self.model_name,
            dimensions=self.dimensions,
            timeout=self.timeout,
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query."""
        return self.get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        embeddings = self.get_text_embeddings([text])
        return embeddings[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API."""
        start_time = time.time()

        logger.debug(
            "Sending embedding request to Qwen OpenAI endpoint",
            api_base=self.api_base,
            model=self.model_name,
            batch_size=len(texts),
            total_chars=sum(len(text) for text in texts),
        )

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float",
                dimensions=self.dimensions,
            )

            embeddings = [item.embedding for item in response.data]
            duration = time.time() - start_time

            logger.info(
                "Qwen OpenAI embedding request completed",
                model=self.model_name,
                batch_size=len(texts),
                response_time_ms=round(duration * 1000, 2),
                embedding_dimensions=len(embeddings[0]) if embeddings else 0,
                total_tokens=getattr(response, "usage", {}).get("total_tokens", 0),
            )

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.dimensions:
                logger.warning(
                    "Unexpected embedding dimensions from Qwen OpenAI API",
                    expected=self.dimensions,
                    actual=len(embeddings[0]),
                    model=self.model_name,
                )

            return embeddings

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Qwen OpenAI embedding request failed",
                error=str(e),
                error_type=type(e).__name__,
                api_base=self.api_base,
                model=self.model_name,
                batch_size=len(texts),
                response_time_ms=round(duration * 1000, 2),
            )
            raise

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query asynchronously (sync fallback)."""
        return self._get_query_embedding(query)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts asynchronously (sync fallback)."""
        return self._get_text_embeddings(texts)

    @classmethod
    def class_name(cls) -> str:
        return "QwenOpenAIEmbedding"


def create_qwen_openai_embedding() -> QwenOpenAIEmbedding:
    """Factory function to create Qwen OpenAI embedding client from settings."""
    return QwenOpenAIEmbedding(
        api_base=settings.QWEN_ENDPOINT or "http://localhost:18080",
        api_key="qwen-embedding-key",
        model_name="qwen3-embedding-0.6b",
        dimensions=1024,  # Qwen3-Embedding-0.6B default
        timeout=getattr(settings, "QWEN_TIMEOUT", 120),
        batch_size=getattr(settings, "QWEN_BATCH_SIZE", 32),
    )
