from __future__ import annotations

import asyncio
import time

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.request_tracking import get_request_tracker

logger = get_logger(__name__)


class Qwen3Embedding(BaseEmbedding):
    """Qwen3 Embedding provider that works with local embedding service."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        service_endpoint: str | None = None,
        dimensions: int = 1024,
        timeout: int = 120,
        batch_size: int = 32,
        **kwargs,
    ):
        """Initialize Qwen3 embedding provider.

        Args:
            model_name: The model name to use
            service_endpoint: URL of the embedding service (e.g., http://embedding-service:8080)
            dimensions: Embedding dimensions
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
        """
        super().__init__(**kwargs)
        self._model_name = model_name
        self._service_endpoint = service_endpoint or settings.EMBEDDING_SERVICE_ENDPOINT
        self._dimensions = dimensions
        self._timeout = timeout
        self._batch_size = batch_size
        self._logger = get_logger(f"{__name__}.qwen3")
        self._tracker = get_request_tracker()

        # Validate configuration
        if not self._service_endpoint:
            raise ValueError(
                "Qwen3 embedding service endpoint not configured. "
                "Set EMBEDDING_SERVICE_ENDPOINT in environment variables."
            )

        # Initialize HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            limits=httpx.Limits(max_keepalive=20, max_connections=100),
        )

        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to the embedding service."""
        try:
            # Simple health check
            response = httpx.get(f"{self._service_endpoint}/health", timeout=10)
            if response.status_code == 200:
                self._logger.info(
                    "Qwen3 embedding service connection successful",
                    endpoint=self._service_endpoint,
                    model=self._model_name,
                )
            else:
                self._logger.warning(
                    "Qwen3 embedding service health check failed",
                    endpoint=self._service_endpoint,
                    status_code=response.status_code,
                )
        except Exception as e:
            self._logger.warning(
                "Could not connect to Qwen3 embedding service",
                endpoint=self._service_endpoint,
                error=str(e),
            )

    async def _get_embeddings_async(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings asynchronously from the service."""
        start_time = time.time()

        try:
            # Prepare request payload
            payload = {"input": texts, "model": self._model_name, "encoding_format": "float"}

            # Add instruction if available
            if hasattr(settings, "EMBEDDING_INSTRUCTION") and settings.EMBEDDING_INSTRUCTION:
                payload["instruction"] = settings.EMBEDDING_INSTRUCTION

            # Make request to embedding service
            response = await self._client.post(f"{self._service_endpoint}/embeddings", json=payload)
            response.raise_for_status()

            # Parse response
            result = response.json()

            if "data" not in result:
                raise ValueError(f"Invalid response format: {result}")

            # Extract embeddings in the correct order
            embeddings = []
            for item in result["data"]:
                embeddings.append(item["embedding"])

            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.info(
                "Qwen3 embedding API batch request completed",
                model=self._model_name,
                batch_size=len(texts),
                total_input_size_chars=sum(len(text) for text in texts),
                response_time_ms=duration_ms,
                output_dimensions=len(embeddings[0]) if embeddings and embeddings[0] else 0,
            )

            return embeddings

        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            self._logger.error(
                "Qwen3 embedding API request failed",
                model=self._model_name,
                batch_size=len(texts),
                response_time_ms=duration_ms,
                error=str(e),
            )
            raise

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        # Use asyncio.run for synchronous calls
        return asyncio.run(self._aget_text_embedding(text))

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text asynchronously."""
        embeddings = await self._get_embeddings_async([text])
        return embeddings[0]

    def _get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        return asyncio.run(self._aget_text_embedding_batch(texts))

    async def _aget_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts asynchronously."""
        # Process in batches if needed
        if len(texts) <= self._batch_size:
            return await self._get_embeddings_async(texts)

        # Process in chunks
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = await self._get_embeddings_async(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars = 1 token average for English)."""
        return len(text) // 4

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup in the running loop
                asyncio.create_task(self._client.aclose())
            else:
                loop.run_until_complete(self._client.aclose())
        except Exception:
            pass  # Ignore cleanup errors


class Qwen3EmbeddingLocal(BaseEmbedding):
    """Qwen3 Embedding provider using local SentenceTransformer model."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto", **kwargs
    ):
        """Initialize local Qwen3 embedding provider.

        Args:
            model_name: The model name to use
            device: Device to use for inference ('auto', 'cpu', 'cuda')
        """
        super().__init__(**kwargs)
        self._model_name = model_name
        self._device = device
        self._logger = get_logger(f"{__name__}.qwen3_local")
        self._tracker = get_request_tracker()

        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._logger.info(
                "Qwen3 local embedding model loaded", model=model_name, device=self._model.device
            )
        except Exception as e:
            self._logger.error(
                "Failed to load Qwen3 local embedding model", model=model_name, error=str(e)
            )
            raise

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        start_time = time.time()

        try:
            embedding = self._model.encode(text, convert_to_list=True)

            duration_ms = round((time.time() - start_time) * 1000, 2)
            self._logger.info(
                "Qwen3 local embedding completed",
                model=self._model_name,
                input_size_chars=len(text),
                response_time_ms=duration_ms,
                output_dimensions=len(embedding) if embedding else 0,
            )

            return embedding
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            self._logger.error(
                "Qwen3 local embedding failed",
                model=self._model_name,
                input_size_chars=len(text),
                response_time_ms=duration_ms,
                error=str(e),
            )
            raise

    def _get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        start_time = time.time()

        try:
            embeddings = self._model.encode(texts, convert_to_list=True)

            duration_ms = round((time.time() - start_time) * 1000, 2)
            self._logger.info(
                "Qwen3 local embedding batch completed",
                model=self._model_name,
                batch_size=len(texts),
                total_input_size_chars=sum(len(text) for text in texts),
                response_time_ms=duration_ms,
                output_dimensions=len(embeddings[0]) if embeddings and embeddings[0] else 0,
            )

            return embeddings
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            self._logger.error(
                "Qwen3 local embedding batch failed",
                model=self._model_name,
                batch_size=len(texts),
                total_input_size_chars=sum(len(text) for text in texts),
                response_time_ms=duration_ms,
                error=str(e),
            )
            raise
