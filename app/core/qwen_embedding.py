"""
Qwen3-Embedding provider implementation using HTTP endpoint.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class QwenEmbedding(BaseEmbedding):
    """Qwen3-Embedding implementation using HTTP endpoint."""

    endpoint_url: str
    model_name: str = "Qwen3-Embedding-0.6B"
    dimensions: int = 1024
    timeout: int = 120
    max_retries: int = 3
    batch_size: int = 32

    _client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = "Qwen3-Embedding-0.6B",
        dimensions: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, embed_batch_size=batch_size, **kwargs)

        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Initialize HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
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
        import asyncio

        # Use asyncio to run the async method
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._aget_text_embeddings(texts))

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async implementation for getting text embeddings."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        import asyncio

        payload = {"inputs": texts, "truncate": True, "normalize": True}

        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Sending embedding request to Qwen endpoint",
                    endpoint=self.endpoint_url,
                    batch_size=len(texts),
                    attempt=attempt + 1,
                    total_chars=sum(len(text) for text in texts),
                )

                response = await self._client.post(
                    f"{self.endpoint_url}/embed", json=payload, headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    embeddings = result.get("embeddings", [])

                    duration = time.time() - start_time

                    logger.info(
                        "Qwen embedding request completed",
                        batch_size=len(texts),
                        response_time_ms=round(duration * 1000, 2),
                        embedding_dimensions=len(embeddings[0]) if embeddings else 0,
                        attempt=attempt + 1,
                    )

                    # Validate dimensions
                    if embeddings and len(embeddings[0]) != self.dimensions:
                        logger.warning(
                            "Unexpected embedding dimensions from Qwen",
                            expected=self.dimensions,
                            actual=len(embeddings[0]),
                        )

                    return embeddings

                else:
                    error_msg = f"Qwen API returned status {response.status_code}: {response.text}"
                    logger.warning(
                        "Qwen embedding request failed",
                        status_code=response.status_code,
                        response_text=response.text[:500],
                        attempt=attempt + 1,
                    )

                    if attempt == self.max_retries - 1:
                        raise RuntimeError(error_msg)

                    # Wait before retry
                    await asyncio.sleep(2**attempt)

            except httpx.RequestError as e:
                error_msg = f"HTTP request error: {str(e)}"
                logger.error(
                    "Qwen embedding HTTP error",
                    error=str(e),
                    endpoint=self.endpoint_url,
                    attempt=attempt + 1,
                )

                if attempt == self.max_retries - 1:
                    raise RuntimeError(error_msg) from e

                # Wait before retry
                await asyncio.sleep(2**attempt)

            except Exception as e:
                error_msg = f"Unexpected error during Qwen embedding: {str(e)}"
                logger.error(
                    "Qwen embedding unexpected error",
                    error=str(e),
                    error_type=type(e).__name__,
                    attempt=attempt + 1,
                )

                if attempt == self.max_retries - 1:
                    raise RuntimeError(error_msg) from e

                # Wait before retry
                await asyncio.sleep(2**attempt)

        raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts")

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query asynchronously."""
        embeddings = await self._aget_text_embeddings([query])
        return embeddings[0]

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "_client"):
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.aclose())
                else:
                    loop.run_until_complete(self._client.aclose())
            except Exception:
                pass  # Ignore cleanup errors

    @classmethod
    def class_name(cls) -> str:
        return "QwenEmbedding"
